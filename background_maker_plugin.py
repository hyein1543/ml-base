from dotenv import load_dotenv
import os
import json
import mimetypes
import glob
import time
from rembg import remove, new_session
from google import genai
from google.genai import types


class BackgroundMakerPlugin:
    def __init__(
        self,
        story_key="fishbread_ep",
        run_tag="4", # for test
        result_suffix="5", # for test
        model_name="gemini-3-pro-image-preview",
        max_reference_images=5, # 읽어올 수 있는 최대 사진 수(이미지 생성의 경우 5장까지가 모델이 효과적으로 참고할 수 있는 최대치로 보임)
    ):
        load_dotenv(override=True)

        self.story_key = story_key
        self.run_tag = run_tag
        self.model_name = model_name
        self.max_reference_images = max_reference_images

        self.input_dir = f"test_input/{self.story_key}"
        self.reference_image_dir = f"{self.input_dir}/pictures"
        self.result_dir = f"result/{self.story_key}{result_suffix}" if result_suffix else f"result/{self.story_key}"
        os.makedirs(self.result_dir, exist_ok=True)

        self.story_file_path = f"{self.input_dir}/{self.story_key}.txt"
        self.elements_json_path = f"{self.input_dir}/{self.story_key}_elements.json"

        self.first_generated_image_path = f"{self.result_dir}/{self.run_tag}.png"
        self.object_only_image_path = f"{self.result_dir}/{self.run_tag}_object_only.png"
        self.background_only_image_path = f"{self.result_dir}/{self.run_tag}_background_only.png"
        self.object_only_cutout_image_path = f"{self.result_dir}/{self.run_tag}_object_only_cutout.png"

        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.chat = self.client.chats.create(
            model=self.model_name,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                tools=[{"google_search": {}}],
            ),
        )

        # 출력 해상도와 비율 ("16:9" & "2K" 조건이 2784x1536 해상도와 가장 유사)
        self.aspect_ratio = "16:9" # "1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9"
        self.resolution = "1K" # "1K", "2K", "4K"

        # rebg 제거 params
        self.rembg_session = new_session("isnet-general-use")
        self.rembg_kwargs = {
            "session": self.rembg_session,
            "alpha_matting": True,
            "alpha_matting_foreground_threshold": 235,
            "alpha_matting_background_threshold": 8,
            "alpha_matting_erode_size": 3,
            "post_process_mask": True,
        }

    @staticmethod
    def _normalize_core_objects(core_objects):
        normalized_core_objects = []
        if isinstance(core_objects, list):
            for obj in core_objects:
                obj_text = str(obj).strip()
                if obj_text:
                    normalized_core_objects.append(obj_text)
        elif isinstance(core_objects, str):
            split_objects = [x.strip() for x in core_objects.split(",")] if "," in core_objects else [core_objects.strip()]
            normalized_core_objects = [x for x in split_objects if x]
        else:
            obj_text = str(core_objects).strip()
            if obj_text:
                normalized_core_objects = [obj_text]
        return normalized_core_objects

    def _collect_reference_image_paths(self):
        reference_image_paths = []
        for ext in ["jpg", "jpeg", "png", "webp"]:
            reference_image_paths.extend(glob.glob(f"{self.reference_image_dir}/*.{ext}"))
            reference_image_paths.extend(glob.glob(f"{self.reference_image_dir}/*.{ext.upper()}"))
        reference_image_paths = sorted(set(reference_image_paths))
        return reference_image_paths[: self.max_reference_images]

    @staticmethod
    def _extract_inline_image(response):
        for part in response.parts:
            if part.text is not None:
                print(part.text)
            elif part.inline_data is not None:
                return part.inline_data.data, part.inline_data.mime_type
        return None, None

    def run(self):
        ### 시간 측정 시작
        script_start_time = time.perf_counter()

        ### 스토리 읽어오기
        with open(self.story_file_path, "r", encoding="utf-8") as f:
            story_text = f.read().strip()

        ### JSON에서 요소 읽어오기
        with open(self.elements_json_path, "r", encoding="utf-8") as f:
            extracted_elements = json.load(f)

        place = extracted_elements.get("장소/공간") or extracted_elements.get("place") or extracted_elements.get("location") or ""
        time_weather = extracted_elements.get("시간대/날씨") or extracted_elements.get("time_weather") or extracted_elements.get("weather") or ""
        mood_keywords = extracted_elements.get("분위기 키워드") or extracted_elements.get("mood_keywords") or extracted_elements.get("mood") or []
        core_objects = extracted_elements.get("핵심 오브젝트") or extracted_elements.get("core_objects") or extracted_elements.get("objects") or []

        if isinstance(mood_keywords, list):
            mood_keywords_text = ", ".join(str(x) for x in mood_keywords)
        else:
            mood_keywords_text = str(mood_keywords)

        normalized_core_objects = self._normalize_core_objects(core_objects)
        core_objects_text = ", ".join(normalized_core_objects)

        # =================== 첫번째 이미지 생성 요청 ==================#
        message = f""" 다음 스토리를 바탕으로, 게임 'To the Moon' 감성의 2D 픽셀아트 JRPG 배경 이미지를 생성해줘.
            참고 사진(들)이 제공되면, 스토리에 등장하는 사물이 사진에 존재할 때 반드시 사진의 사물을 참고해 같은 화풍으로 재드로잉해야 한다.

            [입력]
            - 스토리:
            {story_text}

            [사전 추출 요소 - 최우선 반영]
            아래 요소는 별도 파이프라인에서 추출한 결과야. 결과 이미지에서 “눈에 보이게” 반영해.
            - 장소/공간: {place}
            - 시간대/날씨: {time_weather}
            - 분위기 키워드: {mood_keywords_text}
            - 핵심 오브젝트: {core_objects_text}

            [스토리 반영 규칙]
            - 위 사전 추출 요소를 우선 사용하되, 스토리 본문과 충돌하지 않게 맥락/디테일을 자연스럽게 보강해.
            - 즉, 요소는 JSON 기준으로 고정하고, 장면의 서사적 디테일은 스토리에서 반영해.

            [가장 중요한 규칙: 사진 기반 오브젝트 강제(환각 금지)]
            1) 참고 사진이 없는 경우:
            - 사전 추출 요소와 스토리를 기준으로만 장면을 구성해.
            2) 참고 사진이 있고, 해당 사진에 핵심 오브젝트(1~5개)와 일치하는(매칭 되는) 사물이 있는 경우:
            - 절대 상상으로 디자인하지 말 것.
            - 반드시 사진에 있는 그 사물의 형태/비율/디테일을 근거로 픽셀아트로 재드로잉할 것.
            - 단, 사진의 전체 구도/프레이밍/배경을 복사하지 말고 “사물만” 추출해 사용할 것.
            - 여러 사진에 같은 사물이 있으면 가장 식별이 잘 되는 사진을 우선 참조하고, 부족한 면은 다른 사진으로 보완해.

            [출력 요구사항]
            - 결과물: 투명 배경 금지(완전한 배경), Phaser에서 바로 사용 가능
            - 스타일: 부드러운 16비트 JRPG 픽셀아트, RPG Maker XP style aesthetic, 따뜻한 색감, 약간 흐릿한(감성적 헤이즈) 분위기
            - 조명: 부드럽고 감성적, 먼 곳은 약간 흐릿하게(거리감/헤이즈)
            - 카메라: 오버헤드 뷰(위에서 내려다보는 시점)에서 자연스럽게 보이도록 구성
            - 이동 공간 확보: 캐릭터가 걸어다닐 수 있는 바닥 공간을 충분히 남길 것(화면 하단 35~45%는 비교적 비워두기)
            - 소품/디테일: 가로등, 벤치, 간판 등 분위기 소품 배치(단, 이동 공간을 막지 말 것)

            [캐릭터 규칙]
            - 스토리에 등장하는 인물은 작은 오버헤드 스프라이트로 포함 가능.
            - 단, “나(1인칭 화자)”는 절대 그리지 말 것. (스토리에서 ‘우리/팀’이 등장해도 화자를 제외한 인원만 배치. 주인공 표시/하이라이트 금지)

            [사진 구도 복사 금지]
            - 참고 사진의 카메라 각도, 배치, 프레이밍을 그대로 따라 하지 말 것.
            - 사진은 ‘사물의 디자인/디테일 참고’ 용도이며, 배경 구성은 스토리 기반으로 새로 설계할 것.

            [금지 사항]
            - 실사/3D/벡터/일러스트 느낌 금지
            - 사진 전체를 그대로 재현(구도 복제) 금지
            - 워터마크/로고/서명/깨진 긴 텍스트 금지
            - 과한 번짐/과한 블룸으로 가독성 떨어지는 표현 금지
            - 말풍선 금지"""

        contents = [message]
        # 참조 이미지 모두 contents에 추가
        for reference_image_path in self._collect_reference_image_paths():
            mime_type, _ = mimetypes.guess_type(reference_image_path)
            with open(reference_image_path, "rb") as f:
                image_bytes = f.read()
            contents.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type or "image/jpeg",
                )
            )

        # 첫번째 이미지 생성
        response = self.chat.send_message(
            contents,
            config=types.GenerateContentConfig(
                image_config=types.ImageConfig(
                    aspect_ratio=self.aspect_ratio,
                    image_size=self.resolution,
                ),
            ),
        )

        first_image_bytes, first_image_mime_type = self._extract_inline_image(response)
        if not first_image_bytes or not first_image_mime_type:
            raise RuntimeError("첫번째 이미지 생성에 실패했습니다.")

        with open(self.first_generated_image_path, "wb") as f:
            f.write(first_image_bytes)
        
        # =================== 핵심 오브젝트만 남기기 ==================#
        core_objects_list_text = ", ".join(f"'{obj}'" for obj in normalized_core_objects)
        object_only_message = (
            f"이 이미지에서 다음 핵심 오브젝트만 남겨줘: {core_objects_list_text}. "
            "핵심 오브젝트가 아닌 다른 사물과 배경은 모두 제거하고, "
            "남은 영역은 아무런 패턴이나 그림자가 없는 깨끗한 단색 흰색으로 바꿔줘. "
            "남겨진 핵심 오브젝트들의 크기와 위치는 원본과 동일하게 유지해줘."
        )
        object_response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                object_only_message,
                types.Part.from_bytes(data=first_image_bytes, mime_type=first_image_mime_type),
            ],
            config=types.GenerateContentConfig(
                image_config=types.ImageConfig(
                    aspect_ratio=self.aspect_ratio,
                    image_size=self.resolution,
                ),
            ),
        )

        object_only_bytes, _ = self._extract_inline_image(object_response)
        if not object_only_bytes:
            raise RuntimeError("핵심 오브젝트 분리에 실패했습니다.")

        with open(self.object_only_image_path, "wb") as f:
            f.write(object_only_bytes)

        cutout_bytes = remove(object_only_bytes, **self.rembg_kwargs)
        with open(self.object_only_cutout_image_path, "wb") as f:
            f.write(cutout_bytes)

        # =================== 배경만 남기기 (핵심 오브젝트 제거) ==================#
        background_only_message = (
            f"이 이미지에서 다음 핵심 오브젝트를 제거해줘: {core_objects_list_text}. "
            "핵심 오브젝트를 제외한 다른 사물과 하늘/땅/길/벽면/원경 지형 같은 배경 요소는 유지해줘. "
            "사물이 있던 빈 자리는 주변 맥락에 맞게 자연스럽게 메꿔서, 완전한 배경 이미지로 만들어줘."
        )
        background_response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                background_only_message,
                types.Part.from_bytes(data=first_image_bytes, mime_type=first_image_mime_type),
            ],
            config=types.GenerateContentConfig(
                image_config=types.ImageConfig(
                    aspect_ratio=self.aspect_ratio,
                    image_size=self.resolution,
                ),
            ),
        )

        background_only_bytes, _ = self._extract_inline_image(background_response)
        if not background_only_bytes:
            raise RuntimeError("배경 분리에 실패했습니다.")

        with open(self.background_only_image_path, "wb") as f:
            f.write(background_only_bytes)

        elapsed_seconds = time.perf_counter() - script_start_time
        print(f"[TIME] total elapsed: {elapsed_seconds:.2f}s")


def main():
    plugin = BackgroundMakerPlugin()
    plugin.run()


if __name__ == "__main__":
    main()

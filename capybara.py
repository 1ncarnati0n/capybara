import os
import gc
from typing import List, Union
import time
from datetime import timedelta
import logging
import warnings
import platform
import shutil
import zipfile

import chardet
import ebooklib
from ebooklib import epub
from langdetect import detect
import nltk
from vllm import LLM, SamplingParams
from bs4 import BeautifulSoup
import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch

logging.getLogger().disabled = True
logging.raiseExceptions = False
warnings.filterwarnings('ignore')
nltk.download('punkt', quiet=True)

PathType = Union[str, os.PathLike]

class Capybara:
    def __init__(self, max_len: int = 2048):
        self.max_len = max_len
        self.selected_files = []

        self.upload_msg = None
        self.origin_lang_str = None
        self.target_lang_str = None
        self.origin_lang = None
        self.target_lang = None

        self.upload_files = None

        # 모델 타입 선택 (llm 또는 s2s)
        self.model_type = "llm"  # 기본값: LLM

        # vLLM 모델 설정
        self.selected_model = "davidkim205/iris-7b"
        self.llm = None
        self.sampling_params = None
        self.loaded_model_name = None  # 현재 로드된 vLLM 모델명

        # Seq2Seq 모델 설정
        self.s2s_model = None
        self.s2s_tokenizer = None
        self.src_lang = None
        self.tgt_lang = None

        self.output_folder = 'outputs'
        self.temp_folder_1 = 'temp_1'
        self.temp_folder_2 = 'temp_2'
        self.css = """
            .radio-group .wrap {
                display: float !important;
                grid-template-columns: 1fr 1fr;
            }
            .log-box {
                max-height: 300px;
                overflow-y: auto;
                font-family: monospace;
                font-size: 12px;
            }
            """
        self.start = None
        self.platform = platform.system()

        # 번역 로그 히스토리
        self.translation_logs = []

    def remove_folder(self, temp_folder: PathType):
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)

    def reset_translation_state(self):
        """번역 완료 후 상태 초기화 (모델은 유지)"""
        self.selected_files = []
        self.origin_lang_str = None
        self.target_lang_str = None
        self.origin_lang = None
        self.target_lang = None

        # Seq2Seq 모델의 언어 설정만 초기화 (모델 자체는 유지)
        self.src_lang = None
        self.tgt_lang = None

        # 임시 폴더 정리
        self.remove_folder(self.temp_folder_1)
        self.remove_folder(self.temp_folder_2)

    def add_translation_log(self, model_name: str, duration: str, file_count: int, lang_direction: str):
        """번역 로그 추가"""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {model_name} | {lang_direction} | {file_count}개 파일 | {duration}"

        self.translation_logs.append(log_entry)

        # 최대 50개까지만 유지
        if len(self.translation_logs) > 50:
            self.translation_logs = self.translation_logs[-50:]

        return "\n".join(self.translation_logs)

    def get_translation_logs(self):
        """현재 번역 로그 반환"""
        if not self.translation_logs:
            return "아직 번역 기록이 없습니다."
        return "\n".join(self.translation_logs)

    def main(self):
        self.remove_folder(self.temp_folder_1)
        self.remove_folder(self.temp_folder_2)

        with gr.Blocks(
            css=self.css,
            title='Capybara',
            theme=gr.themes.Default(primary_hue="blue", secondary_hue="cyan")
        ) as app:
            gr.HTML("<div align='center'><h1 style='margin-top:10px;'>AI 한영/영한 번역기 <span style='color:blue'>카피바라</span></h1><p style='color:gray;'>LLM + Seq2Seq 듀얼 모델 지원</p></div>")

            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    with gr.Tab('순서 1'):
                        gr.Markdown("<h3>1. 번역할 파일들 선택</h3>")
                        input_window = gr.File(
                            file_count="multiple",
                            file_types=[".txt", ".epub", ".srt"],
                            label='파일들'
                        )
                        lang_msg = gr.HTML(self.upload_msg)
                        input_window.change(
                            fn=self.change_upload,
                            inputs=input_window,
                            outputs=lang_msg,
                            preprocess=False
                        )

                with gr.Column(scale=2):
                    with gr.Tab('순서 2'):
                        gr.Markdown("<h3>2. 번역 모델 선택</h3>")
                        model_selector = gr.Radio(
                            choices=[
                                ("LLM (iris-7b) - 빠르고 자연스러운 번역 ⚡", "llm"),
                                ("Seq2Seq (NLLB) - 안정적이고 정확한 번역 💡", "s2s")
                            ],
                            value="llm",
                            label="번역 모델",
                            elem_classes="radio-group"
                        )

                        translate_btn = gr.Button(
                            value="번역 실행하기",
                            size='lg',
                            variant="primary",
                            interactive=True
                        )

                        gr.HTML("<div style='text-align:right'><p style='color:grey;'>처음 실행 시 모델을 다운받는데 시간이 걸립니다.</p><p style='color:orange;'>LLM: 5-10배 빠른 속도, 자연스러운 문체</p><p style='color:green;'>Seq2Seq: 안정적 번역, 낮은 VRAM</p></div>")

                        with gr.Row():
                            status_msg = gr.Textbox(
                                label="현재 상태",
                                scale=4,
                                value='번역 대기 중...'
                            )

                            btn_openfolder = gr.Button(
                                value='📂 번역 완료한 파일들 보기',
                                scale=1,
                                variant="secondary"
                            )
                            btn_openfolder.click(
                                fn=lambda: self.open_folder(),
                                inputs=None,
                                outputs=None
                            )

                        # 번역 로그 히스토리
                        gr.Markdown("<h3>📜 번역 히스토리</h3>")
                        log_display = gr.Textbox(
                            label="번역 기록",
                            value="",
                            lines=10,
                            max_lines=15,
                            interactive=False,
                            elem_classes="log-box"
                        )

                        translate_btn.click(
                            fn=self.translateFn,
                            inputs=model_selector,
                            outputs=[status_msg, log_display]
                        )

        app.queue().launch(
            inbrowser=True,
            allowed_paths=["."]
        )

    def finalize_fn(self) -> str:
        sec = self.check_time()
        self.start = None
        return sec

    def get_vllm_model(self):
        """vLLM 모델 초기화/교체"""
        # models 폴더 생성
        models_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(models_dir, exist_ok=True)

        need_reload = (self.llm is None) or (self.loaded_model_name != self.selected_model)

        if need_reload:
            if self.llm is not None and self.loaded_model_name != self.selected_model:
                print(f"vLLM 모델 변경: {self.loaded_model_name} → {self.selected_model}")
                # 기존 엔진 해제 및 캐시 정리
                self.llm = None
                try:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

            print(f"vLLM 모델 로딩 중: {self.selected_model}")

            self.llm = LLM(
                model=self.selected_model,
                download_dir=models_dir,  # models 폴더에 다운로드
                tensor_parallel_size=1,
                gpu_memory_utilization=0.91,
                max_model_len=1024,
                dtype="auto",
                kv_cache_dtype="fp8",
                enforce_eager=True,
                trust_remote_code=True
            )

            # 샘플링 파라미터 설정 (번역 품질 최적화)
            self.sampling_params = SamplingParams(
                temperature=0.3,
                top_p=0.95,
                max_tokens=128,
                repetition_penalty=1.1,
                skip_special_tokens=True,
                stop=["\n", "English:", "Korean:", "---", "###"]
            )

            self.loaded_model_name = self.selected_model
            print("vLLM 모델 로딩 완료!")

        return self.llm

    

    def get_seq2seq_model(self):
        """Seq2Seq (NLLB) 모델 초기화"""
        if self.s2s_model is None:
            print(f"Seq2Seq 모델 로딩 중...")

            # models 폴더 생성
            models_dir = os.path.join(os.getcwd(), "models")
            os.makedirs(models_dir, exist_ok=True)

            # 언어별 모델 선택
            if self.origin_lang == "en":
                model_name = "NHNDQ/nllb-finetuned-en2ko"
                self.src_lang = "eng_Latn"
                self.tgt_lang = "kor_Hang"
            else:
                model_name = "NHNDQ/nllb-finetuned-ko2en"
                self.src_lang = "kor_Hang"
                self.tgt_lang = "eng_Latn"

            print(f"모델: {model_name}")

            # 모델 및 토크나이저 로드
            self.s2s_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                cache_dir=models_dir
            )
            self.s2s_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=models_dir,
                src_lang=self.src_lang,
                tgt_lang=self.tgt_lang
            )

            # GPU 사용 가능하면 GPU로 이동
            if torch.cuda.is_available():
                self.s2s_model = self.s2s_model.cuda()

            print("Seq2Seq 모델 로딩 완료!")

        return self.s2s_model

    def translate_text(self, text: str) -> str:
        """단일 텍스트 번역 (모델 타입에 따라 다른 로직 사용)"""
        if not text.strip():
            return ""

        if self.model_type == "llm":
            # LLM: domain_fewshot 전략 (prompt_tuner.py 검증 결과 적용)
            if self.origin_lang == "en":
                prompt = f"""You are a professional translator.
Preserve tone and nuances.
Output only the Korean translation without quotes or labels.

Examples:
English: "Good morning."
Korean: "좋은 아침입니다."

English: "Thank you very much."
Korean: "정말 감사합니다."

Now translate this:
English: "{text}"
Korean:"""
            else:
                prompt = f"""You are a professional translator.
Preserve tone and nuances.
Output only the English translation without quotes or labels.

Examples:
Korean: "좋은 아침입니다."
English: "Good morning."

Korean: "정말 감사합니다."
English: "Thank you very much."

Now translate this:
Korean: "{text}"
English:"""

            # vLLM 추론
            outputs = self.llm.generate([prompt], self.sampling_params)
            translated = outputs[0].outputs[0].text.strip()

            # 후처리: 불필요한 따옴표 제거
            translated = translated.strip('"').strip("'").strip()

        else:  # s2s
            # Seq2Seq: 직접 모델 사용
            inputs = self.s2s_tokenizer(text, return_tensors="pt", padding=True)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            outputs = self.s2s_model.generate(**inputs, max_length=self.max_len)
            translated = self.s2s_tokenizer.decode(outputs[0], skip_special_tokens=True)

        return translated

    def translate_batch(self, texts: List[str]) -> List[str]:
        """배치 번역 (모델 타입에 따라 다른 로직 사용)"""
        if not texts:
            return []

        # 빈 텍스트 필터링
        non_empty_indices = [i for i, text in enumerate(texts) if text.strip()]
        non_empty_texts = [texts[i] for i in non_empty_indices]

        if not non_empty_texts:
            return [""] * len(texts)

        if self.model_type == "llm":
            # LLM: domain_fewshot 전략 (prompt_tuner.py 검증 결과 적용)
            if self.origin_lang == "en":
                prompts = [f"""You are a professional translator.
Preserve tone and nuances.
Output only the Korean translation without quotes or labels.

Examples:
English: "Good morning."
Korean: "좋은 아침입니다."

English: "Thank you very much."
Korean: "정말 감사합니다."

Now translate this:
English: "{text}"
Korean:""" for text in non_empty_texts]
            else:
                prompts = [f"""You are a professional translator.
Preserve tone and nuances.
Output only the English translation without quotes or labels.

Examples:
Korean: "좋은 아침입니다."
English: "Good morning."

Korean: "정말 감사합니다."
English: "Thank you very much."

Now translate this:
Korean: "{text}"
English:""" for text in non_empty_texts]

            # vLLM 배치 추론
            outputs = self.llm.generate(prompts, self.sampling_params)
            translated_non_empty = [
                output.outputs[0].text.strip().strip('"').strip("'").strip()  # 후처리 추가
                for output in outputs
            ]

        else:  # s2s
            # Seq2Seq: 배치 번역
            inputs = self.s2s_tokenizer(non_empty_texts, return_tensors="pt", padding=True, truncation=True)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            outputs = self.s2s_model.generate(**inputs, max_length=self.max_len)
            translated_non_empty = [
                self.s2s_tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]

        # 결과 재구성 (빈 텍스트 위치 복원)
        result = [""] * len(texts)
        for i, translated in zip(non_empty_indices, translated_non_empty):
            result[i] = translated

        return result

    def translateFn(self, model_type: str, progress=gr.Progress()):
        if not self.selected_files:
            current_log = self.get_translation_logs()
            return "번역할 파일을 추가하세요.", current_log

        # 모델 타입 설정
        self.model_type = model_type

        self.start = time.time()

        # 모델별 초기화
        if self.model_type == "llm":
            progress(0, desc="LLM 번역 모델을 준비중입니다...")
            self.get_vllm_model()
        else:
            progress(0, desc="Seq2Seq 번역 모델을 준비중입니다...")
            self.get_seq2seq_model()

        origin_abb = self.origin_lang
        target_abb = "ko" if self.origin_lang == "en" else "en"

        # 파일 개수 기록
        file_count = len(self.selected_files)

        for file in progress.tqdm(self.selected_files, desc='파일로딩'):
            name, ext = os.path.splitext(file['orig_name'])

            if 'epub' in ext:
                self.translate_epub(file, name, ext, origin_abb, target_abb, progress)

            elif 'srt' in ext:
                self.translate_srt(file, name, ext, origin_abb, target_abb, progress)

            else:  # txt
                self.translate_txt(file, name, ext, origin_abb, target_abb, progress)

        sec = self.finalize_fn()
        model_name = "LLM (iris-7b)" if self.model_type == "llm" else "Seq2Seq (NLLB)"

        # 번역 방향 문자열 생성
        lang_direction = f"{self.origin_lang_str} → {self.target_lang_str}"

        # 로그 추가
        updated_log = self.add_translation_log(model_name, sec, file_count, lang_direction)

        # 상태 초기화 (다음 번역을 위해)
        self.reset_translation_state()

        status_message = f"✅ 번역 완료! 모델: {model_name}, 걸린 시간: {sec}"

        return status_message, updated_log

    def translate_txt(self, file, name, ext, origin_abb, target_abb, progress):
        """TXT 파일 번역"""
        output_file_1, output_file_2, book = self.get_file_info(origin_abb, target_abb, name, ext, file)

        book_list = book.read().split(sep='\n')

        for paragraph in progress.tqdm(book_list, desc='단락'):
            if not paragraph.strip():
                output_file_1.write('\n')
                output_file_2.write('\n')
                continue

            # 문장 분리
            sentences = nltk.sent_tokenize(paragraph)

            # 배치 번역
            translated_sentences = self.translate_batch(sentences)

            # 파일 작성
            for orig, trans in zip(sentences, translated_sentences):
                output_file_1.write(f"{trans} ({orig}) ")
                output_file_2.write(f'{trans} ')

            output_file_1.write('\n')
            output_file_2.write('\n')

        output_file_1.close()
        output_file_2.close()
        book.close()

    def translate_srt(self, file, name, ext, origin_abb, target_abb, progress):
        """SRT 파일 번역"""
        output_file_1, output_file_2, book = self.get_file_info(origin_abb, target_abb, name, ext, file)
        srt_list = self.get_srt_list(book.read())

        # 모든 자막 텍스트 추출
        texts = [item['text'] for item in srt_list]

        # 배치 번역
        translated_texts = self.translate_batch(texts)

        # SRT 형식으로 작성
        for item, translated in progress.tqdm(zip(srt_list, translated_texts), desc='자막', total=len(srt_list)):
            translated = translated.replace('.', '')  # SRT는 마침표 제거
            output_file_1.write(f"{item['num']}\n{item['time']}\n{translated} ({item['text']})\n\n")
            output_file_2.write(f"{item['num']}\n{item['time']}\n{translated}\n\n")

        output_file_1.close()
        output_file_2.close()
        book.close()

    def translate_epub(self, file, name, ext, origin_abb, target_abb, progress):
        """EPUB 파일 번역"""
        self.zip_extract(self.temp_folder_1, file['path'])
        self.zip_extract(self.temp_folder_2, file['path'])

        file_path = self.get_html_list()

        for html_file in progress.tqdm(file_path, desc='챕터'):
            html_file_2 = html_file.replace(self.temp_folder_1, self.temp_folder_2)

            input_file_1 = open(html_file, 'r', encoding='utf-8')
            input_file_2 = open(html_file_2, 'r', encoding='utf-8')

            soup_1 = BeautifulSoup(input_file_1.read(), 'html.parser')
            soup_2 = BeautifulSoup(input_file_2.read(), 'html.parser')

            p_tags_1 = soup_1.find_all('p')
            p_tags_2 = soup_2.find_all('p')

            # p 태그가 없으면 div 태그로 대체
            if not p_tags_1:
                p_tags_1 = soup_1.find_all('div')
                p_tags_2 = soup_2.find_all('div')
                for p_tag_1, p_tag_2 in zip(p_tags_1, p_tags_2):
                    if not p_tag_1.find('div'):
                        if p_tag_1.text.strip():
                            p_tag_1.name = 'p'
                            p_tag_2.name = 'p'

                p_tags_1 = soup_1.find_all('p')
                p_tags_2 = soup_2.find_all('p')

            # 각 단락 처리
            for text_node_1, text_node_2 in progress.tqdm(zip(p_tags_1, p_tags_2), desc='단락 수', total=len(p_tags_1)):
                if not text_node_1.text.strip():
                    continue

                p_tag_1 = soup_1.new_tag('p')
                p_tag_2 = soup_2.new_tag('p')

                try:
                    if text_node_1.attrs and text_node_1.attrs.get('class'):
                        p_tag_1['class'] = text_node_1.attrs['class']
                        p_tag_2['class'] = text_node_1.attrs['class']
                except:
                    pass

                # 문장 분리
                sentences = nltk.sent_tokenize(text_node_1.text)

                # 배치 번역
                translated_sentences = self.translate_batch(sentences)

                # 번역 결과 조합
                combined_1 = ' '.join([f"{trans} ({orig})" for orig, trans in zip(sentences, translated_sentences)])
                combined_2 = ' '.join(translated_sentences)

                p_tag_1.string = combined_1
                p_tag_2.string = combined_2

                # 이미지 태그 유지
                img_tag = text_node_1.find('img')
                if img_tag:
                    p_tag_1.append(img_tag)
                    p_tag_2.append(img_tag)

                text_node_1.replace_with(p_tag_1)
                text_node_2.replace_with(p_tag_2)

            input_file_1.close()
            input_file_2.close()

            output_file_1 = open(html_file, 'w', encoding='utf-8')
            output_file_2 = open(html_file_2, 'w', encoding='utf-8')

            output_file_1.write(str(soup_1))
            output_file_2.write(str(soup_2))
            output_file_1.close()
            output_file_2.close()

        # EPUB 재패키징
        for loc_folder in [self.temp_folder_1, self.temp_folder_2]:
            self.zip_folder(loc_folder, f'{loc_folder}.epub')

        os.makedirs(self.output_folder, exist_ok=True)

        # 모델 타입 접미사 추가
        model_suffix = "llm" if self.model_type == "llm" else "s2s"

        shutil.move(
            f'{self.temp_folder_1}.epub',
            os.path.join(self.output_folder, f"{name}_{target_abb}({origin_abb})_{model_suffix}{ext}")
        )
        shutil.move(
            f'{self.temp_folder_2}.epub',
            os.path.join(self.output_folder, f"{name}_{target_abb}_{model_suffix}{ext}")
        )

        self.remove_folder(self.temp_folder_1)
        self.remove_folder(self.temp_folder_2)

    def get_srt_list(self, srt_file):
        """SRT 파일 파싱"""
        srt_list_raw = srt_file.strip().split('\n')
        len_srt = len(srt_list_raw)

        srt_list = []
        recent_num = 0
        for len_idx in range(0, len_srt, 4):
            if len_idx + 2 >= len_srt or not srt_list_raw[len_idx+2].strip():
                continue
            recent_num += 1
            srt_list.append({
                'num': recent_num,
                'time': srt_list_raw[len_idx+1],
                'text': srt_list_raw[len_idx+2].strip(),
            })
        return srt_list

    def change_upload(self, files: List):
        """파일 업로드 시 언어 감지"""
        try:
            self.selected_files = files
            if not files:
                return self.upload_msg

            aBook = files[0]
            name, ext = os.path.splitext(aBook['path'])

            if '.epub' in ext:
                file = epub.read_epub(aBook['path'])
                lang = file.get_metadata('DC', 'language')
                if lang:
                    check_lang = lang[0][0]
                else:
                    # EPUB 내용에서 언어 감지
                    for item in file.get_items():
                        if item.get_type() == ebooklib.ITEM_DOCUMENT:
                            soup = BeautifulSoup(item.get_body_content(), 'html.parser')
                            all_tags = soup.find_all('p')
                            if not all_tags:
                                continue

                            text_tags = [tag.text for tag in all_tags if tag.text.strip()]
                            lang_str = ' '.join(text_tags)
                            check_lang = detect(lang_str[0:500])
                            if 'en' in check_lang or 'ko' in check_lang:
                                break
                            else:
                                return "<p style='text-align:center;color:red;'>표준 규격을 벗어난 epub입니다.</p>"

            elif '.srt' in ext:
                srt_file = self.get_filename(aBook['path'], ext)
                srt_list = self.get_srt_list(srt_file.read())
                srt_texts = ' '.join([srt['text'] for srt in srt_list[:50]])
                check_lang = detect(srt_texts[0:200])
                srt_file.close()
            else:
                book = self.get_filename(aBook['path'], ext)
                check_lang = detect(book.read()[0:200])
                book.close()

            self.origin_lang_str = '영어' if 'en' in check_lang else "한국어"
            self.target_lang_str = '한국어' if 'en' in check_lang else "영어"
            self.origin_lang = "en" if 'en' in check_lang else "ko"
            self.target_lang = "ko" if 'en' in check_lang else "en"

            return f"<p style='text-align:center;'><span style='color:skyblue;font-size:1.5em;'>{self.origin_lang_str}</span><span>를 </span> <span style='color:red;font-size:1.5em;'> {self.target_lang_str}</span><span>로 번역합니다.</span></p>"
        except Exception as err:
            print(f"언어 감지 오류: {err}")
            return "<p style='text-align:center;color:red;'>어떤 언어인지 알아내는데 실패했습니다.</p>"

    def get_filename(self, file_name, ext):
        """파일 인코딩 감지 및 열기"""
        try:
            if '.srt' in ext:
                encoding = 'utf-8'
            else:
                check_encoding = open(file_name, 'rb')
                result = chardet.detect(check_encoding.read(10000))
                encoding = result['encoding']
                check_encoding.close()
            input_file = open(file_name, 'r', encoding=encoding)
            return input_file
        except Exception as err:
            print(err)
            return None

    def get_file_info(self, origin_abb, target_abb, name, ext, file):
        """출력 파일 정보 생성"""
        # 모델 타입 접미사 추가
        model_suffix = "llm" if self.model_type == "llm" else "s2s"

        output_file_1 = self.write_filename(f"{name}_{target_abb}({origin_abb})_{model_suffix}{ext}")
        output_file_2 = self.write_filename(f"{name}_{target_abb}_{model_suffix}{ext}")
        book = self.get_filename(file['path'], ext)
        return output_file_1, output_file_2, book

    def write_filename(self, file_name: str):
        """출력 파일 생성"""
        saveDir = self.output_folder
        if not os.path.isdir(saveDir):
            os.makedirs(saveDir)

        file = os.path.join(saveDir, file_name)
        output_file = open(file, 'w', encoding='utf-8')
        return output_file

    def open_folder(self):
        """출력 폴더 열기"""
        saveDir = self.output_folder
        command_to_open = ''

        if not os.path.isdir(saveDir):
            os.makedirs(saveDir)

        if self.platform == 'Windows':
            command_to_open = f"start {saveDir}"
        elif self.platform == 'Darwin':
            command_to_open = f"open {saveDir}"
        elif self.platform == 'Linux':
            command_to_open = f"xdg-open {saveDir}"

        os.system(command_to_open)

    def zip_extract(self, folder_path: PathType, epub_file: PathType):
        """EPUB 압축 해제"""
        try:
            zip_module = zipfile.ZipFile(epub_file, 'r')
            os.makedirs(folder_path, exist_ok=True)
            zip_module.extractall(folder_path)
            zip_module.close()
        except:
            print('잘못된 epub파일입니다')

    def zip_folder(self, folder_path: PathType, epub_name: PathType):
        """폴더를 EPUB으로 압축"""
        try:
            zip_module = zipfile.ZipFile(epub_name, 'w', zipfile.ZIP_DEFLATED)
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    zip_module.write(file_path, os.path.relpath(file_path, folder_path))
            zip_module.close()
        except Exception as err:
            print('epub 파일을 생성하는데 실패했습니다.')
            print(err)

    def get_html_list(self) -> List:
        """EPUB 내 HTML 파일 목록 가져오기"""
        file_paths = []
        for root, _, files in os.walk(self.temp_folder_1):
            for file in files:
                if file.endswith(('xhtml', 'html', 'htm')):
                    file_paths.append(os.path.join(root, file))
        return file_paths

    def check_time(self) -> str:
        """경과 시간 계산"""
        end = time.time()
        during = end - self.start
        sec = str(timedelta(seconds=during)).split('.')[0]
        return sec

if __name__ == "__main__":
    capybara = Capybara()
    capybara.main()

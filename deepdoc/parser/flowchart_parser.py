import logging

import cv2
import numpy as np
import torch
from PIL import Image
import easyocr
from openai import OpenAI
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from api.db.services.llm_service import LLMBundle
from api.db import LLMType

class FlowchartVisionParser:
    def __init__(self, tenant_id, figures_data, sam_model=None, sam_checkpoint_path=None, easyocr_model_path=None, **kwargs):
        self.figures_data = figures_data
        self._extract_figures_info()
        self.tenant_id = tenant_id

        # 初始化 SAM 模型
        if sam_model and sam_checkpoint_path:
            DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            MODEL_TYPE = sam_model
            self.sam = sam_model_registry[MODEL_TYPE](checkpoint=sam_checkpoint_path).to(device=DEVICE)
            self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        else:
            self.sam = None
            self.mask_generator = None

        # 初始化 OCR
        if easyocr_model_path:
            self.reader = easyocr.Reader(
                ['en', 'ch_sim'],
                model_storage_directory=easyocr_model_path,
                download_enabled=False  # 避免重复下载
            )
        else:
            self.reader = easyocr.Reader(['en', 'ch_sim'])

    def _extract_figures_info(self):
        """
        从 figures_data 中提取图像和相关信息
        """
        self.figures = []
        self.descriptions = []
        self.positions = []
        self.is_flowchart_list = []  # 新增：存储是否为流程图的标识

        for item in self.figures_data:
            if len(item) == 3:  # 现在每个元素有3个部分：图像、位置、是否为流程图
                img_desc, positions, is_flowchart = item
                if isinstance(img_desc, tuple) and len(img_desc) == 2:
                    image, desc = img_desc
                    if isinstance(image, Image.Image):
                        self.figures.append(image)
                        self.descriptions.append(desc)
                        self.positions.append(positions)
                        self.is_flowchart_list.append(is_flowchart)  # 记录是否为流程图

    def _pil_to_cv2(self, pil_image):
        """
        将 PIL 图像转换为 OpenCV 格式
        """
        # 转换为 numpy 数组
        numpy_image = np.array(pil_image)
        # PIL 图像通常是 RGB，OpenCV 使用 BGR
        cv2_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        return cv2_image

    def _segment_image_with_sam(self, image):
        """
        使用 SAM 对图像进行分割
        """
        if not self.sam or not self.mask_generator:
            return None, None

        sam_result = self.mask_generator.generate(image)

        # 提取分割信息
        masks = [
            mask['segmentation']
            for mask in sorted(sam_result, key=lambda x: x['point_coords'][0][1] if x['area'] > 10000 else False)
        ]

        boxes = [
            mask['bbox']
            for mask in sorted(sam_result, key=lambda x: x['point_coords'][0][1] if x['area'] > 10000 else False)
        ]

        return masks, boxes

    def _recognize_text_in_segments(self, image, boxes):
        """
        在分割区域中识别文本
        """
        texts = []
        height, width = image.shape[:2]

        for box in boxes:
            x, y, w, h = map(int, box)

            # 边界检查
            if x >= width or y >= height:
                texts.append("")
                continue
            if x + w > width:
                w = width - x
            if y + h > height:
                h = height - y

            # 提取图像片段
            segment = image[y:y + h, x:x + w]

            # OCR 识别
            try:
                result = self.reader.readtext(segment)

                # 提取唯一文本并过滤低置信度结果
                unique_texts = []
                seen_texts = set()
                for (_, text, confidence) in result:
                    if confidence > 0.3 and text.strip() and text not in seen_texts:
                        unique_texts.append(text.strip())
                        seen_texts.add(text)

                texts.append(' '.join(unique_texts))
            except Exception as e:
                logging.warning(f"OCR failed for segment: {e}")
                texts.append("")

        return texts

    def _analyze_flowchart_with_llm(self, texts, boxes):
        """
        使用 LLM 分析流程图
        """
        client = OpenAI(
            api_key="sk-a7e186bd115845a8b31edae0d4d8f1c8",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        prompt = (
            "Give a detailed and descriptive interpretation in Chinese of the flowchart "
            "in the form of steps using following details OCR text recognition data: "
            f"{texts} bounding box info obtained from sam(segment anything model): {boxes}"
        )

        try:
            # response = client.chat.completions.create(
            #     model="qwen3-14b",
            #     messages=[{"role": "system", "content": prompt}],
            #     max_tokens=1000,
            #     extra_body = {
            #         "enable_thinking": False  # 禁用思考模式
            #     }
            # )
            # return response.choices[0].message.content.replace('\n', '').replace(' .', '.').strip()

            chat_mdl = LLMBundle(self.tenant_id, LLMType.CHAT, "")
            gen_conf = {"temperature": 0.9}
            ans = chat_mdl.chat(
                prompt,
                [{"role": "user", "content": " "}],
                gen_conf,
            )
            # data=[re.sub(r"^[0-9]\. ", "", a) for a in ans.split("\n") if re.match(r"^[0-9]\. ", a)]
            return ans
        except Exception as e:
            return f"Error analyzing flowchart: {str(e)}"

    def __call__(self, **kwargs):
        """
        处理所有流程图并返回增强后的描述
        """
        callback = kwargs.get("callback", lambda prog, msg: None)

        enhanced_figures = []

        for i, (figure, original_desc, position, is_flowchart) in enumerate(zip(
                self.figures, self.descriptions, self.positions, self.is_flowchart_list)):
            try:
                # 如果不是流程图，则跳过处理
                if not is_flowchart:
                    enhanced_figure = ((figure, ""), position)
                    enhanced_figures.append(enhanced_figure)
                    callback((i + 1) / len(self.figures), f"Skipped non-flowchart {i + 1}")
                    continue

                callback(i / len(self.figures) * 0.5, f"Processing flowchart {i + 1}/{len(self.figures)}")

                # 转换图像格式
                cv2_image = self._pil_to_cv2(figure)

                # 使用 SAM 分割（使用原始图像）
                callback(i / len(self.figures) * 0.5 + 0.2, f"Segmenting flowchart {i + 1}")
                masks, boxes = self._segment_image_with_sam(cv2_image)

                if masks is None or boxes is None:
                    # 如果没有 SAM，使用原始描述
                    enhanced_description = "\n".join(original_desc)
                else:
                    # OCR 识别文本（使用原始图像）
                    callback(i / len(self.figures) * 0.5 + 0.3, f"Recognizing text in flowchart {i + 1}")
                    texts = self._recognize_text_in_segments(cv2_image, boxes)

                    # 使用 LLM 分析流程图
                    callback(i / len(self.figures) * 0.5 + 0.4, f"Analyzing flowchart {i + 1}")
                    enhanced_description = self._analyze_flowchart_with_llm(texts, boxes)

                # 组合原始描述和增强描述
                combined_description = enhanced_description + "\n" + "\n".join(original_desc)

                # 构造返回格式，与 VisionFigureParser 兼容
                enhanced_figure = ((figure, [combined_description]), position)
                enhanced_figures.append(enhanced_figure)

                callback((i + 1) / len(self.figures) * 0.5 + 0.5, f"Completed flowchart {i + 1}")

            except Exception as e:
                # 出错时保留原始描述
                enhanced_figure = ((figure, original_desc), position)
                enhanced_figures.append(enhanced_figure)
                callback((i + 1) / len(self.figures) * 0.5 + 0.5, f"Error processing flowchart {i + 1}: {str(e)}")

        return enhanced_figures

# api_detector.py

import cv2
import numpy as np
from ultralytics import YOLO

class StoneDetector:
    """
    Classe para detecção binária de pedras nos rins usando YOLOv8-seg.
    """

    def __init__(self, model_path: str, conf: float = 0.3):
        # Carrega o modelo
        self.model = YOLO(model_path)
        self.conf = conf

        # Mapeia índices e nomes
        self.names = self.model.names  # ex: {0:'DJ',1:'bladder',2:'kidney',3:'stone',4:'ureter'}
        # Procura de forma robusta a classe "stone" (ignora case)
        self.stone_idx = None
        for idx, name in self.names.items():
            if name.lower() == "stone":
                self.stone_idx = idx
                break

        if self.stone_idx is None:
            print("⚠️ Atenção: classe 'stone' não encontrada em model.names:", self.names)

    def predict(self, image_bytes: bytes) -> dict:
        """
        Recebe bytes de imagem, faz segmentação e retorna {'has_stone': True/False}.
        """
        try:
            # Decodifica imagem
            img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return {"error": "Formato de imagem inválido"}

            # Inferência segmentação
            results = self.model.predict(img, conf=self.conf, task='segment')[0]

            # DEBUG: Mostra todas as classes detectadas
            detected_idxs = [int(box.cls) for box in results.boxes] if results.boxes is not None else []
            detected_names = [self.names[i] for i in detected_idxs]
            print(f"[DEBUG] Índices detectados: {detected_idxs}")
            print(f"[DEBUG] Nomes detectados:  {detected_names}")

            # Se não achou o índice de stone, retorna False
            if self.stone_idx is None:
                return {"has_stone": False}

            # Verifica se "stone" está na lista de classes detectadas
            if self.stone_idx in detected_idxs:
                return {"has_stone": True}

            return {"has_stone": False}

        except Exception as e:
            return {"error": f"Erro na predição: {str(e)}"}


if __name__ == "__main__":
    # Teste local
    detector = StoneDetector("best_model.pt", conf=0.15)  # tente conf=0.15 para pedrinhas mais fracas
    with open("exemplo.jpg", "rb") as f:
        img_bytes = f.read()

    resultado = detector.predict(img_bytes)
    print("Resultado binário:", resultado)

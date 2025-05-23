import os
import cv2
import torch
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from ultralytics import YOLO


# VERIFICAÇÃO DO DATASET 
def validar_dataset(dataset_path: str):
    """Valida estrutura e consistência do dataset."""
    def verificar_pasta(split: str):
        img_dir = os.path.join(dataset_path, split, "images")
        lbl_dir = os.path.join(dataset_path, split, "labels")
        
        if not all(os.path.exists(d) for d in [img_dir, lbl_dir]):
            raise FileNotFoundError(f"Pastas ausentes em {split} (images/labels)")

        imagens = {os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith('.jpg')}
        labels = {os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith('.txt')}
        
        if imagens != labels:
            missing = imagens - labels
            print(f"[{split}] {len(missing)} imagens sem labels")
        else:
            print(f"[{split}] Labels correspondem às imagens.")

    for split in ['train', 'valid', 'test']:
        verificar_pasta(split)
    print("Dataset validado com sucesso!\n")


# ANOTAÇÕES 
def visualizar_anotacoes(dataset_path: str, num_exemplos: int = 2):
    """Mostra algumas imagens do dataset com suas máscaras desenhadas."""
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(dataset_path, split, "images")
        lbl_dir = os.path.join(dataset_path, split, "labels")

        imagens = sorted(os.listdir(img_dir))[:num_exemplos]
        for img_file in imagens:
            img_path = os.path.join(img_dir, img_file)
            lbl_path = os.path.join(lbl_dir, img_file.replace('.jpg', '.txt'))

            img = cv2.imread(img_path)
            if img is None:
                print(f"Imagem não encontrada: {img_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)

            try:
                with open(lbl_path, 'r') as f:
                    for line in f.readlines():
                        class_id, *coords = line.strip().split()
                        coords = np.array([float(x) for x in coords], dtype=np.float32).reshape(-1, 2)
                        coords_pixel = (coords * np.array([w, h])).astype(np.int32)
                        cv2.fillPoly(mask, [coords_pixel], color=255)
            except FileNotFoundError:
                print(f"Label não encontrado: {lbl_path}")
                continue

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title(f"{split.upper()} - Imagem")
            plt.imshow(img)
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.title("Máscara")
            plt.imshow(mask, cmap='gray')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"visualizacao_{split}_{img_file}.png")
            plt.close()



# CLASSE DE INFERÊNCIA 
class DetectorPedras:
    """Carrega o modelo treinado e realiza inferência em imagens."""

    def __init__(self, modelo_path: str):
        self.modelo = YOLO(modelo_path)
        self.nomes_classes = self.modelo.names

    def prever(self, imagem: np.ndarray, conf: float = 0.3) -> dict:
        """Executa predição em uma imagem e retorna resultados."""
        resultados = self.modelo.predict(imagem, conf=conf)
        saida = {"boxes": [], "masks": [], "classes": []}

        if hasattr(resultados[0], "masks") and resultados[0].masks is not None:
            for mask_array in resultados[0].masks.data:
                saida["masks"].append(mask_array.cpu().numpy().tolist())

        if resultados[0].boxes is not None:
            for box in resultados[0].boxes:
                saida["boxes"].append(box.xyxy.cpu().numpy().tolist())
                saida["classes"].append(self.nomes_classes[int(box.cls)])

        return saida

    def salvar_resultados(self, imagem: np.ndarray, caminho_saida: str):
        """Salva imagem com anotações desenhadas."""
        resultados = self.modelo.predict(imagem)
        anotada = resultados[0].plot()
        cv2.imwrite(caminho_saida, anotada)
        print(f" Resultado salvo em: {caminho_saida}")


# EXECUÇÃO PRINCIPAL
def main(args):
    validar_dataset(args.dataset)

    if args.visualizar:
        visualizar_anotacoes(args.dataset, num_exemplos=1)

    detector = DetectorPedras(args.model)

    if args.imagem:
        imagem = cv2.imread(args.imagem)
        if imagem is None:
            print(f" Imagem não encontrada: {args.imagem}")
            return

        resultados = detector.prever(imagem)
        print(" Resultado da Inferência:")
        print(resultados)

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            saida_path = os.path.join(args.output_dir, "resultado.jpg")
            detector.salvar_resultados(imagem, saida_path)


# ARGUMENTOS 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=" Detector de Pedras nos Rins com YOLOv8")
    parser.add_argument('--dataset', required=True, help="Caminho para o dataset no formato YOLOv8")
    parser.add_argument('--model', required=True, help="Caminho do modelo .pt treinado")
    parser.add_argument('--imagem', help="Caminho da imagem para fazer inferência")
    parser.add_argument('--output_dir', help="Pasta onde salvar a imagem com predição")
    parser.add_argument('--visualizar', action='store_true', help="Visualizar exemplos do dataset")

    args = parser.parse_args()
    main(args)

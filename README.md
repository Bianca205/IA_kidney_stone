
# 🧠Detecção de Pedras nos Rins com YOLOv8

Este projeto realiza a detecção de pedras nos rins e estruturas do trato urinário (rim, bexiga, ureteres, etc.) a partir de imagens de raio-X, utilizando segmentação com o modelo YOLOv8.
Desenvolvido como trabalho da disciplina de Mineração de dados, o objetivo é treinar, validar e entregar um modelo funcional que possa ser integrado a uma API.

O modelo foi treinado, validado e testado com o dataset **Stone in Urinary System**, disponível no Roboflow.

## 📂 Dataset

- **Fonte**: [Roboflow - Stone in Urinary System](https://universe.roboflow.com/xray-abdomen-to-identify-stones/stone_in_urinary_system)
- **Divisão**:
  - **Treinamento**: 2121 imagens (87%)
  - **Validação**: 203 imagens (8%)
  - **Teste**: 101 imagens (4%)
- **Pré-processamento**:
  - Auto-orientação
  - Resize para 640x640
  - Aumentos: brilho (-5% a 0%), 3 outputs por exemplo.

## 🚀 Modelo

- **Arquitetura**: YOLOv8x-seg (segmentação e detecção)
- **Técnicas utilizadas**:
  - Aumento de dados: `mosaic`, `copy-paste`
  - Mixed Precision (AMP) para otimização de memória
  - Máscara com `overlap_mask` e `mask_ratio=2`
  - Otimização: SGD com `momentum=0.9`

## ⚙️ Treinamento do modelo utilizado

- **Imagem**: 960x960
- **Epochs**: 100
- **Batch size**: 8 (ajustado para evitar OOM)
- **Pacote**: Ultralytics 8.3.142
- **Hardware**: NVIDIA A100-SXM4-40GB (Pode ser usado T4)

### Exemplo de comando (Modelo-v2: modelo utilizado):

```python
results = model.train(
    data=os.path.join(dataset_path, "data.yaml"),
    imgsz=960,
    epochs=100,
    patience=25,
    batch=8,
    optimizer='SGD',
    lr0=0.01,
    momentum=0.9,
    augment=True,
    scale=0.5,
    translate=0.05,
    hsv_h=0.0,
    erasing=0.0,
    mosaic=0.3,
    copy_paste=0.1,
    single_cls=False,
    overlap_mask=True,
    mask_ratio=2,
    device=0,
    amp=True
)
```

## 📝 Resultados

- **mAP50** geral: ~0.85
- **mAP50-95**: ~0.55
- **Classe Stone**: detecção com confiança baixa (~0.42 mAP), mas funcional.
- Observação: as imagens de teste eram especialmente difíceis, contendo muito ruído e pedra minúsculas.

## 🛠️ Como usar o modelo

1. **Clonar o repositório**.
2. **Instalar dependências**:

```bash
pip install -r requirements.txt
```

3. **Executar a inferência**:

```python
from ultralytics import YOLO

model = YOLO('path_to_your_model.pt')
results = model.predict('path_to_image.jpg', conf=0.25)
```

## 💾 Como salvar o modelo

Após treinamento, o modelo final é salvo automaticamente em:

```
runs/segment/train/weights/best.pt 
```

Pode ser movido para outro diretório com:

```python
import shutil
shutil.copy('runs/segment/train/weights/best.pt', 'detector/stone_detector.pt')
```





## 📁 Estrutura da Pasta Stone_IA 

```bash
Stone_IA/
├── main_detector.py      # Script principal para validação, visualização e execução do modelo
├── api_detector.py       # Classe usada pelo backend para integrar a predição com a API
├── best_model.pt         # Modelo YOLOv8 segmentado treinado (.pt)
├── requirements.txt      # Bibliotecas necessárias para rodar o projeto
├── exemplo.jpg           # Imagem de raio-X usada para teste local (opcional)
└── README.md             # Este arquivo de instruções

*🚀 Como Executar o Script Principal*
Para rodar o script principal com uma imagem de teste:
python main_detector.py \
  --dataset ./dataset \
  --model best_model.pt \
  --imagem exemplo.jpg \
  --output_dir ./saida \
  --visualizar

Parâmetros:

--dataset: caminho da pasta com o dataset YOLOv8 (com pastas train/valid/test)

--model: caminho do modelo treinado (.pt)

--imagem: imagem local para inferência

--output_dir: onde salvar a imagem com a predição

--visualizar: exibe imagens + máscaras do dataset (True/False)

*👨‍🏫 Finalidade Acadêmica*
Este projeto foi desenvolvido como parte da disciplina de Mineração de dados, demonstrando o uso de redes neurais profundas (YOLOv8) para apoio ao diagnóstico por imagem (RAIO-X).
```
# 🖥️ Deploy

O modelo será integrado via **API** pelo time de backend e frontend, possibilitando a inferência via HTTP.

## ✅ Requisitos

Veja o arquivo `requirements.txt` para todas as dependências com versões fixas.

## 📚 Referências

- Ultralytics YOLOv8 Documentation
- Roboflow Datasets
- PyTorch CUDA compatibility matrix

## ✍️ Autor

- Equipe de desenvolvimento: [Gyovana Garcês, Hitawana Silva, Davi Borges].
- Data: 22 de Maio de 2025

## 📥 Instruções de Uso

> **OBS:** O dataset é muito grande para subir no repositório.  
> Baixe-o em: `https://universe.roboflow.com/xray-abdomen-to-identify-stones/stone_in_urinary_system/dataset/7`

1. Descompacte o ZIP na raiz do projeto e renomeie a pasta para `dataset/`

## Modelo
O modelo foi removido do repositório por exceder 100 MB (limite do GitHub), mas está disponível para download aqui:

🔗 [Clique para baixar o modelo treinado (best_model.pt)](https://drive.google.com/file/d/1ObtX8LzbeC3aXLog1l5vewfVaYclcFRD/view?usp=drive_link)

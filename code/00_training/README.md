# Stage 1 Autoencoder Training and Embedding Extraction

이 폴더에는 supplier identification에 사용하는 48차원 embedding을 생성하기 위한 학습 코드, 모델 정의, 데이터 로더가 들어 있다. 학습된 checkpoint는 Git 공유 대상에서 제외한다.

## 전체 흐름

```text
voxel shape + manufacturing metadata
                  |
                  v
        MultiModalAutoencoder
                  |
                  v
          48-dimensional latent z
                  |
                  v
       train/test embedding CSV
                  |
                  v
 Stage 1 supplier identification
```

모델은 3D voxel 형상과 제조 관련 metadata를 동시에 입력받는다. Shape encoder와 metrics encoder의 출력을 결합하여 48차원 latent vector `z`를 만들며, 이 vector가 downstream supplier identification의 입력으로 사용된다.

## 폴더 구성

```text
00_training/
├── 01_train_multimodal_autoencoder.ipynb
├── 02_extract_supplier_embeddings.ipynb
├── multimodal_autoencoder.py
├── voxel_dataset.py
├── binvox_io.py
├── training_config.py
└── checkpoints/                 # 로컬 학습 출력; Git에서는 제외
```

- `01_train_multimodal_autoencoder.ipynb`: multimodal autoencoder 학습 및 checkpoint 저장
- `02_extract_supplier_embeddings.ipynb`: 저장된 checkpoint를 불러와 train/test embedding 생성
- `multimodal_autoencoder.py`: shape encoder/decoder, metrics encoder/decoder 및 `MultiModalAutoencoder` 정의
- `voxel_dataset.py`: embedding 추출 시 `.binvox` 파일을 PyTorch tensor로 변환
- `binvox_io.py`: BINVOX 파일 입출력 유틸리티
- `training_config.py`: epoch, learning rate, batch size 등 기본 학습 설정
- `checkpoints/`: 로컬 학습 시 생성되는 모델 가중치 저장 위치 (`*.pth`는 Git에서 제외)

## 입력 데이터

학습 및 embedding 추출에 사용하는 CSV는 저장소 루트 기준 다음 위치에 있다.

```text
data/01_supplier_identification/main_split_70_30/
├── train_dataset_without_quantity.csv
└── test_dataset_without_quantity.csv
```

코드는 `filename` 또는 `FileName` 열을 사용해 CSV 행과 voxel 파일을 연결한다. 모델에 사용되는 metadata 열은 다음과 같다.

- `LogScaled_Time`
- `LogScaled_Cost`
- `LogScaled_Quantity`
- `LogScaled_Tolerance`
- `LogScaled_Density`
- `LogScaled_Service_Temperature`
- `LogScaled_Ultimate_Tensile`

Embedding 추출 CSV에는 supplier label을 나타내는 `Supplier` 열도 필요하다.

원시 voxel 데이터는 파일 크기 때문에 이 패키지에 포함되어 있지 않다. 현재 notebook은 저장소 루트에서 다음 외부 경로를 사용한다.

```text
../../../GRA/3D_Voxel/Total_Dataset/
```

- 학습 notebook은 위 경로의 `pt_cache/*.pt`를 사용한다.
- Embedding 추출 notebook은 위 경로의 `*.binvox`를 사용한다.

다른 위치에서 voxel 데이터를 사용할 경우 각 notebook의 `voxel_dir` 값을 수정해야 한다.

## 1. Autoencoder 학습

`01_train_multimodal_autoencoder.ipynb`를 실행한다. 주요 기본 설정은 다음과 같다.

| 설정 | 값 |
|---|---:|
| Epochs per run | 10 |
| Batch size | 8 |
| Optimizer | Adam |
| Learning rate | 0.00004 |
| Adam betas | (0.8, 0.99) |
| Latent dimension | 48 |
| Metadata scale | 10.0 |
| Checkpoint interval | 5 epochs |

Quantity 입력은 supplier identification에서 수량의 영향을 제거하기 위해 0으로 고정한다. 형상 복원 loss는 weighted BCE와 Dice loss를 결합하며, time, cost, tolerance 및 material properties의 복원 MSE도 함께 최적화한다.

Checkpoint는 다음 형식으로 저장된다.

```text
checkpoints/multimodal_autoencoder_epoch_NNN.pth
```

예를 들어 epoch 10 checkpoint는 `multimodal_autoencoder_epoch_010.pth`이다. 학습 loss 기록과 TensorBoard log도 `checkpoints/` 아래에 생성된다.

## 2. Embedding 추출

직접 학습했거나 별도로 전달받은 학습모델을 이용할 경우 `02_extract_supplier_embeddings.ipynb`를 실행한다. 기본 설정은 epoch 10 checkpoint를 사용하지만, checkpoint 파일 자체는 이 저장소에서 제공하지 않는다.

```python
RUN_ALL_CHECKPOINTS = False
RUN_SELECTED_EPOCHS = True
SELECTED_EPOCHS = [10]
```

선택할 수 있는 실행 방식은 다음과 같다.

- `RUN_ALL_CHECKPOINTS = True`: `checkpoints/`에 있는 모든 `.pth` 처리
- `RUN_SELECTED_EPOCHS = True`: `SELECTED_EPOCHS`에 지정한 checkpoint 처리
- 두 옵션 모두 `False`: `model_path_single`에 지정한 단일 checkpoint 처리

각 sample에 대해 모델의 latent output `z`를 저장한다. 생성 파일은 다음 위치에 기록된다.

```text
data/01_supplier_identification/main_split_70_30/
├── train_embeddings_epoch_010.csv
└── test_embeddings_epoch_010.csv
```

출력 열은 다음 세 개다.

| 열 | 설명 |
|---|---|
| `filename` | 원본 voxel 파일명 |
| `supplier` | 학습 행의 supplier ID 또는 test 행의 feasible supplier 목록 |
| `embedding` | 48차원 latent vector |

논문 identification 분석에는 metadata가 결합된 다음 파일을 사용한다.

```text
train_embeddings_epoch_010_with_metadata.csv
test_embeddings_epoch_010_with_metadata.csv
```

## 3. Supplier identification 실행

Embedding 생성이 끝나면 다음 notebook으로 이동한다.

```text
../01_supplier_identification/01_evaluate_supplier_identification.ipynb
../01_supplier_identification/02_tune_neighbor_threshold.ipynb
```

`01_evaluate_supplier_identification.ipynb`는 baseline 1, baseline 2 및 proposed embedding-based method를 비교한다. `02_tune_neighbor_threshold.ipynb`는 neighbor threshold `K`에 따른 identification 성능을 평가한다.

## 실행 환경

주요 Python dependency는 다음과 같다.

```text
torch
pandas
numpy
tqdm
tensorboard
jupyter
```

Notebook은 저장소 루트 또는 `code/00_training` 폴더에서 실행하도록 구성되어 있다. GPU가 있으면 CUDA를 사용하고, 그렇지 않으면 CPU를 사용한다.

## 주의사항

- 학습된 checkpoint(`*.pth`)는 공유 대상이 아니며 `.gitignore`로 제외된다.
- 전체 voxel 데이터가 없으면 기존 embedding CSV를 이용한 identification 평가는 가능하지만, 재학습이나 embedding 재생성은 불가능하다.
- 직접 생성하거나 별도로 전달받은 `.pth` 파일은 `multimodal_autoencoder.py`의 `MultiModalAutoencoder` 구조와 함께 사용해야 한다.

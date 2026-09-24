# GalaxEye Backend Engineer — ML Systems
## Design Note

### 1. Approach

I designed the system as a small offline image-classification service.

The main flow is:

Satellite tile
    ↓
FastAPI `/predict` endpoint
    ↓
Image validation and preprocessing
    ↓
Local ResNet18 classifier
    ↓
Prediction + confidence
    ↓
Confidence check
    ↓
SQLite
    ↓
Analyst query through `/predictions`

The system does not depend on any hosted model API or internet connection during inference. The trained model is stored locally as `satellite_model.pth`.

For the take-home implementation, I used ResNet18 with transfer learning. The provided candidate dataset contains seven classes:

- AnnualCrop
- Forest
- Highway
- Industrial
- Residential
- River
- SeaLake

I froze the pretrained ResNet18 layers and trained the final classification layer on the provided dataset. This keeps training lightweight enough to run on CPU.

### 2. Tile flow

When a tile is uploaded to `/predict`, the service first checks that the uploaded file is an image.

The image is converted to RGB and resized to 224x224 pixels. The same normalization used during training is applied before inference.

The local model returns class probabilities. I store the highest-probability class as the prediction and its probability as the confidence.

I also added a simple confidence policy:

- confidence >= 0.60 → `accepted`
- confidence < 0.60 → `uncertain`

The threshold is an initial operational choice, not a claim that 0.60 is an optimal threshold. In a production system I would select it using validation data and the cost of different types of errors.

### 3. Storage

I used SQLite because the assignment is an offline system and the working slice does not require a distributed database.

Each prediction stores:

- filename
- predicted class
- confidence
- status
- model version
- creation timestamp

The model version is stored with each prediction so that results can be traced back to the model that generated them.

For a larger deployment, I would consider PostgreSQL or another embedded/local database depending on the hardware, concurrency requirements and expected data volume.

### 4. Querying

The API exposes:

`GET /predictions`

to retrieve stored results.

It also supports filtering by predicted class and minimum confidence, for example:

`GET /predictions?prediction=Forest`

and:

`GET /predictions?min_confidence=0.8`

This gives an analyst a simple way to inspect predictions without directly accessing the database.

For a larger system, I would consider additional filters such as time range, model version and uncertainty status.

### 5. Model confidence and incorrect predictions

I would not treat every model prediction as equally reliable.

Low-confidence predictions are stored as `uncertain` instead of being silently treated as trusted results. This preserves the prediction for later review while making the uncertainty visible to the analyst.

The evaluation results also showed that overall accuracy does not tell the whole story. The model achieved 80% accuracy on the provided 210-image evaluation set, but performance varied significantly by class. For example, Highway and Industrial achieved 96.67%, while River achieved 46.67% and SeaLake 60%.

This suggests that the next model improvement should investigate class-specific errors rather than only optimizing overall accuracy.

### 6. Trade-offs

#### SQLite vs a larger database

SQLite:
- Simple
- Offline
- No separate database service
- Easy to deploy on isolated hardware

A larger database would be preferable if concurrent writes, multiple services or large-scale querying became important.

#### Pretrained model vs training from scratch

I chose transfer learning because it provides a useful baseline with limited training time and CPU requirements.

Training from scratch would provide more control but would require more data, compute and experimentation.

#### Accept every prediction vs confidence threshold

Accepting every prediction is simpler but hides model uncertainty.

The confidence threshold adds a small amount of system logic and makes uncertain results visible to analysts.

The threshold would need calibration before production use.

### 7. Assumptions

I assumed:

- Tiles are standard image files.
- Each tile represents one primary land-use class.
- The provided candidate dataset is representative enough for a take-home baseline.
- CPU inference is acceptable for the thin slice.
- SQLite is sufficient for the expected local workload.
- Analysts primarily need class, confidence and timestamp information.

### 8. Questions I would ask GalaxEye

If I could clarify the production requirements, I would ask:

1. What satellite sensors and image formats will the production system receive?
2. Are tiles always RGB, or can they contain multispectral bands?
3. What inference latency and throughput are required?
4. How much storage is available on the isolated hardware?
5. What should happen to low-confidence predictions operationally?
6. Do analysts need spatial metadata such as latitude, longitude or tile coordinates?
7. How frequently will the model be updated?
8. Is there a human review workflow for uncertain predictions?
9. What level of auditability is required for model predictions?
10. What hardware will the offline service run on?

### 9. What I would improve for production

The current implementation is intentionally small.

For production I would consider:

- model calibration and better confidence estimation
- class-level monitoring
- confusion-matrix based evaluation
- model version management
- structured application logs
- health and readiness checks
- batch inference for higher throughput
- retry/error handling around ingestion
- database backups
- model integrity checks
- monitoring for data drift and changes in image distribution


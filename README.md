# Satellite Ship Detection (Tiled CNN + Clustering, served via FastAPI)

Detects and counts container ships in satellite imagery. The API accepts a satellite image **plus its ground resolution** and returns estimated ship locations and a total count.

**Key idea:** classify overlapping tiles with a CNN, then **deduplicate detections** by clustering positive tiles (DBSCAN) and reporting **cluster centroids** as ship positions.

![/analyze/image API endpoint output.](misc/performance.png)

## Input / Output

- **Input:** a `.png` satellite image + resolution (m/pixel)
- **Output:**
  - `ship_count`
  - `ships`: list of `(x, y)` ship positions in image coordinates (centroids of clustered detections)
  - (optional) an annotated image with predicted locations (example image)

## Approach

- **Resolution normalization:** images are rescaled to a consistent target (e.g., 3.0 m/pixel) so tile size corresponds to a stable ground area.
- **Overlapping tiling:** increases recall for ships near tile boundaries.
- **DBSCAN deduplication:** converts many tile-level hits into a single ship location by clustering nearby positives.

## Data

Trained/evaluated using Planet’s openly licensed **Open California** dataset (San Francisco Bay area), distributed as “Ships in Satellite Imagery” on Kaggle.  
Dataset: https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery

## Model + Inference

- **Tile classifier:** small CNN (stacked Conv + MaxPool blocks)
- **Tile prediction:** binary ship / no-ship with confidence score
- **Post-processing:** DBSCAN over positive tile centers → cluster centroids → ship locations

## Evaluation (current)

These are the metrics currently tracked in this repo:

- **Training accuracy:** 98% with class balance 1:4 for positives to negatives.
- **Real landscape imagery recall:** **0.90**
  - Observed failure mode: **false positives near narrow docks / dock-like linear structures**

## API

### Endpoints

- `POST /api/v1/analyze/json` → JSON ship count + positions
- `POST /api/v1/analyze/image` → returns the original image with detections drawn

### Example response (shape)

```json
{
  "ship_count": 3,
  "ships": [
    {"x": 512, "y": 284},
    {"x": 1044, "y": 901},
    {"x": 188, "y": 732}
  ]
}
```

## Run locally

```bash
uvicorn app.main:app --reload
```

Open interactive docs:
- http://127.0.0.1:8000/docs

## Notes / Known limitations

- Performance depends on how similar inputs are to the training domain (San Francisco Bay imagery).
- Dock-like structures can trigger false positives; this is a known error mode of the current pipeline.


# Jalanlytics - Agent Implementation Guide

This document provides comprehensive instructions for Claude agents to understand, extend, and improve the Jalanlytics codebase.

## Project Overview

**Jalanlytics** is a street income intelligence tool for Indonesian markets. It analyzes CCTV footage to estimate area income levels by detecting, tracking, and classifying vehicles.

### Key Insight
Indonesian streets have a unique vehicle profile:
- **Motorcycles dominate**: 70-80% of traffic, ~90 million registered in Indonesia
- **High motorcycle density**: Dozens per frame, often overlapping
- **Chinese brands growing**: Wuling, DFSK, Chery, BYD gaining market share
- **Commercial vehicles**: Ojol (Gojek, Grab, Shopee) are ubiquitous

### Target Users
Business development professionals looking to identify optimal retail, service, or real estate locations.

---

## Architecture Overview

### Pipeline Flow
```
Video Input
    ↓
Frame Sampling (VideoProcessor)
    ↓
Vehicle Detection (YOLOv8)
    ↓
Vehicle Tracking (ByteTrack)
    ↓
Commercial Detection (Color + Pattern)
    ↓
Vehicle Classification (CLIP)
    ↓
Price Lookup (YAML Database)
    ↓
Report Generation (Rich/JSON/CSV)
```

### Module Responsibilities

| Module | File | Purpose |
|--------|------|---------|
| VideoProcessor | `src/video/processor.py` | Load videos, sample frames, handle formats |
| VehicleDetector | `src/detection/detector.py` | YOLOv8 wrapper for vehicle detection |
| CommercialVehicleDetector | `src/detection/commercial.py` | Identify ojol, angkot, taxis |
| VehicleTracker | `src/tracking/tracker.py` | ByteTrack for unique vehicle counting |
| CLIPClassifier | `src/classification/clip_classifier.py` | Zero-shot make/model classification |
| VehicleClassifier | `src/classification/classifier.py` | Ensemble classifier interface |
| PriceDatabase | `src/data/prices.py` | Indonesian vehicle price lookups |
| ReportGenerator | `src/report/generator.py` | Format analysis results |
| AnalysisPipeline | `src/pipeline.py` | Orchestrate all components |

---

## Critical Technical Details

### YOLO Class IDs (COCO)
```python
BICYCLE = 1
CAR = 2
MOTORCYCLE = 3
BUS = 5
TRUCK = 7
```

### Frame Sampling
- Default: Every 10th frame (configurable)
- For 30fps video: ~3 samples/second
- Balances accuracy vs processing speed

### Tracking (ByteTrack)
```python
track_activation_threshold = 0.25  # Min confidence to start track
lost_track_buffer = 30            # Frames to keep lost tracks
minimum_matching_threshold = 0.8   # IoU for matching
```

### Classification (CLIP)
- Model: `openai/clip-vit-base-patch32`
- Zero-shot with predefined labels
- Labels in `src/classification/clip_classifier.py`

---

## Indonesian Vehicle Data

### Price Tiers (Cars - IDR)
| Tier | Range | Examples |
|------|-------|----------|
| Budget | < 150 juta | Wuling Air EV, Daihatsu Ayla |
| Economy | 150-300 juta | Toyota Avanza, Suzuki Ertiga |
| Mid-range | 300-500 juta | Toyota Innova, Honda HR-V |
| Premium | 500-800 juta | Toyota Fortuner, Honda CR-V |
| Luxury | > 800 juta | Toyota Alphard, BMW, Mercedes |

### Price Tiers (Motorcycles - IDR)
| Tier | Range | Examples |
|------|-------|----------|
| Budget | < 18 juta | Honda Beat, Yamaha Mio |
| Mid | 18-35 juta | Honda Vario, Yamaha NMAX |
| Premium | > 35 juta | Honda PCX, Yamaha R15 |

### Market Share (Approximate)
**Motorcycles:**
- Honda: ~75% (Beat, Vario, PCX, Scoopy)
- Yamaha: ~20% (NMAX, Aerox, Mio)
- Others: ~5%

**Cars:**
- Toyota: ~30%
- Daihatsu: ~15%
- Honda: ~12%
- Mitsubishi: ~10%
- Chinese brands (Wuling, DFSK, Chery): Growing fast

### Commercial Vehicle Indicators

| Type | Indicators |
|------|------------|
| Gojek | Green uniform/jacket, green delivery box |
| Grab | Green uniform, green helmet |
| Shopee Food | Orange uniform, orange delivery box |
| Maxim | Red uniform |
| Angkot | Minibus shape, often colored by route |
| Blue Bird Taxi | Blue color |

---

## CLI Interface

```bash
# Basic analysis
jalanlytics analyze video.mp4

# With options
jalanlytics analyze video.mp4 \
  --output report.json \
  --format json \
  --frame-interval 5 \
  --confidence 0.3 \
  --device cuda

# Generate config file
jalanlytics init-config

# Check GPU status
jalanlytics check-gpu

# List available models
jalanlytics list-models
```

### CLI Options
| Option | Default | Description |
|--------|---------|-------------|
| `--output, -o` | stdout | Output file path |
| `--format, -f` | text | text/json/csv |
| `--frame-interval, -i` | 10 | Process every Nth frame |
| `--confidence, -c` | 0.25 | Min detection confidence |
| `--device, -d` | auto | cuda/cpu/auto |
| `--no-commercial` | false | Disable commercial detection |
| `--config` | - | Path to YAML config |
| `--verbose, -v` | false | Enable verbose output |

---

## Report Output Format

### Text Report Structure
```
=== Street Income Analysis Report ===

Video: [filename]
Duration: [MM:SS]
Frames analyzed: [N]

VEHICLE COUNTS
--------------
Motorcycles detected: X unique vehicles
  (Commercial/Ojol: Y)
Cars detected: X unique vehicles
Total: N

MOTORCYCLE BREAKDOWN
--------------------
Honda Beat: N (X%)
[Top 10 models]

Motorcycle tier distribution:
  Budget: X%
  Mid: X%
  Premium: X%

CAR BREAKDOWN
-------------
[Same structure as motorcycles]

INCOME ANALYSIS
---------------
Median motorcycle value: Rp X juta
Median car value: Rp X juta
Motorcycle-to-car ratio: X:1

AREA ASSESSMENT
---------------
Estimated income bracket: [BRACKET]
Characteristics: [Description]
Good location for:
  - [Recommendation 1]
  - [Recommendation 2]
```

### Income Brackets
- HIGH-INCOME
- UPPER-MIDDLE-INCOME
- MIDDLE-INCOME
- LOWER-MIDDLE-INCOME
- LOW-INCOME

---

## Implementation Guidelines

### Code Style
- Python 3.10+ with type hints
- Docstrings for all public functions/classes
- Use dataclasses for data containers
- Follow existing patterns in the codebase

### Testing Requirements
- Unit tests for each module in `tests/`
- Test with sample videos of varying quality
- Test GPU and CPU inference paths

### Error Handling
- Graceful degradation (CPU fallback if no GPU)
- Informative error messages
- Don't crash on single frame failures

---

## Known Challenges & Solutions

### 1. Dense Motorcycle Clusters
**Problem:** Indonesian streets can have 20+ motorcycles in one frame, overlapping.

**Solution:**
- Lower confidence threshold (0.25)
- ByteTrack handles occlusion well
- Accept some missed detections

### 2. Similar-Looking Scooters
**Problem:** Honda Beat vs Yamaha Mio look nearly identical.

**Solution:**
- Return top-3 predictions
- Use tier-based analysis (both are budget tier)
- Future: Train custom classifier

### 3. Chinese Brand Recognition
**Problem:** CLIP has limited knowledge of Wuling, DFSK, etc.

**Solution:**
- Include explicit labels in CLIP prompts
- Use shape-based classification as fallback
- Future: Fine-tune on Indonesian vehicle dataset

### 4. Commercial Vehicle Detection
**Problem:** Need to identify ojol without uniform ML training.

**Solution:**
- Color detection in HSV space
- Look for delivery boxes via edge detection
- Buses/trucks are always commercial

### 5. Varying Video Quality
**Problem:** CCTV footage ranges from 240p to 1080p.

**Solution:**
- No forced resize (let YOLO handle it)
- Adjust confidence based on quality
- Sample more frames for low-res

---

## Extension Points

### Adding New Vehicle Labels
Edit `src/classification/clip_classifier.py`:
```python
MOTORCYCLE_LABELS = [
    # Add new labels here
    "New Brand Model motorcycle",
]
```

### Adding New Commercial Detectors
Edit `src/detection/commercial.py`:
```python
# Add new color ranges
NEW_SERVICE_RANGES = [
    ((H_low, S_low, V_low), (H_high, S_high, V_high)),
]
```

### Adding New Price Data
Edit `src/data/vehicle_prices.yaml`:
```yaml
motorcycles:
  new_brand:
    new_model:
      price: 25000000
      tier: mid
      category: scooter
```

### Adding Custom Classifier
Extend `BaseClassifier` in `src/classification/base.py`:
```python
class CustomClassifier(BaseClassifier):
    def classify(self, image, vehicle_class):
        # Your implementation
        pass
```

---

## Future Enhancements

### Priority 1 (High Impact)
1. **Custom Indonesian Vehicle Dataset**
   - Scrape OLX/Tokopedia Otomotif
   - Include Wuling, DFSK, local motorcycle variants
   - Train fine-tuned classifier

2. **Real Vehicle Crop Classification**
   - Currently using placeholder classification
   - Need to store frame references during tracking
   - Classify actual vehicle crops

3. **Visualization Output**
   - Annotated video output
   - Use supervision annotators
   - Bounding boxes with labels

### Priority 2 (Medium Impact)
4. **Batch Processing**
   - Process multiple videos
   - Aggregate results
   - Generate comparative reports

5. **Confidence Calibration**
   - Adjust thresholds based on video quality
   - Auto-detect optimal settings

6. **License Plate Region Detection**
   - Can help with make/model via database
   - Privacy considerations

### Priority 3 (Nice to Have)
7. **Web Interface**
   - Upload video
   - View results in browser
   - Export reports

8. **Real-time Streaming**
   - RTSP input support
   - Live analysis dashboard

---

## Quick Start for Development

```bash
# Clone and setup
cd jalanlytics
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -e .

# Run analysis
jalanlytics analyze sample_video.mp4

# Run with verbose
jalanlytics analyze sample_video.mp4 -v

# Generate config
jalanlytics init-config
```

---

## File Structure Reference

```
jalanlytics/
├── src/
│   ├── __init__.py
│   ├── main.py                 # CLI entry point
│   ├── config.py               # Configuration dataclasses
│   ├── pipeline.py             # Main orchestrator
│   ├── video/
│   │   ├── __init__.py
│   │   └── processor.py        # Video loading/sampling
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── detector.py         # YOLOv8 wrapper
│   │   └── commercial.py       # Commercial vehicle detection
│   ├── tracking/
│   │   ├── __init__.py
│   │   └── tracker.py          # ByteTrack wrapper
│   ├── classification/
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract classifier
│   │   ├── clip_classifier.py  # CLIP implementation
│   │   └── classifier.py       # Ensemble interface
│   ├── data/
│   │   ├── __init__.py
│   │   ├── prices.py           # Price lookup logic
│   │   └── vehicle_prices.yaml # Price database
│   └── report/
│       ├── __init__.py
│       └── generator.py        # Report formatting
├── tests/
├── sample_data/
├── requirements.txt
├── pyproject.toml
└── AGENT_INSTRUCTIONS.md       # This file
```

---

## Contact & Resources

- **Price Data Sources:**
  - OLX Autos Indonesia
  - Tokopedia Otomotif
  - Manufacturer websites

- **Training Data Sources:**
  - OLX vehicle listings (scrape images)
  - Instagram Indonesian car/motorcycle accounts
  - YouTube Indonesian traffic videos

---

*Last Updated: December 2024*

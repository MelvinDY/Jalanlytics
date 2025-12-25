# Jalanlytics

**Street Income Intelligence for Indonesian Markets**

Analyze CCTV footage of Indonesian streets to estimate area income levels by detecting, tracking, and classifying vehicles.

## Features

- **Vehicle Detection**: YOLOv8-powered detection of motorcycles, cars, bicycles, buses, and trucks
- **Unique Vehicle Tracking**: ByteTrack prevents double-counting the same vehicle
- **Indonesian Focus**: Optimized for high motorcycle density (70-80% of traffic)
- **Commercial Detection**: Identifies ojol (Gojek, Grab, Shopee), angkot, and taxis
- **CLIP Classification**: Zero-shot vehicle make/model identification
- **Income Analysis**: Maps vehicles to price tiers and estimates area income level
- **Multiple Output Formats**: Text, JSON, and CSV reports

## Installation

### Requirements
- Python 3.10+
- CUDA-compatible GPU (recommended) or CPU

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/jalanlytics.git
cd jalanlytics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .
```

### GPU Support (Recommended)

For CUDA support, install PyTorch with CUDA:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### Basic Analysis

```bash
jalanlytics analyze street_footage.mp4
```

### With Options

```bash
jalanlytics analyze video.mp4 \
  --output report.json \
  --format json \
  --frame-interval 5 \
  --device cuda
```

### Python API

```python
from src.pipeline import AnalysisPipeline
from src.config import Config

config = Config()
pipeline = AnalysisPipeline(config)
result = pipeline.analyze("street_video.mp4")
report = pipeline.generate_report(result)
print(report)
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `jalanlytics analyze <video>` | Analyze a video file |
| `jalanlytics init-config` | Generate default config file |
| `jalanlytics check-gpu` | Check GPU availability |
| `jalanlytics list-models` | List available models |

### Analysis Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output, -o` | stdout | Output file path |
| `--format, -f` | text | Output format (text/json/csv) |
| `--frame-interval, -i` | 10 | Process every Nth frame |
| `--confidence, -c` | 0.25 | Min detection confidence |
| `--device, -d` | auto | Device (cuda/cpu/auto) |
| `--no-commercial` | false | Disable commercial detection |

## Sample Output

```
=== Street Income Analysis Report ===

Video: jalan_sudirman.mp4
Duration: 2:34
Frames analyzed: 150

VEHICLE COUNTS
--------------
Motorcycles detected: 156 unique vehicles
  (Commercial/Ojol: 23)
Cars detected: 34 unique vehicles
Total: 193

MOTORCYCLE BREAKDOWN
--------------------
Honda Beat: 42 (27%)
Honda Vario: 28 (18%)
Yamaha NMAX: 24 (15%)
...

INCOME ANALYSIS
---------------
Median motorcycle value: Rp 22 juta
Median car value: Rp 280 juta
Motorcycle-to-car ratio: 4.6:1

AREA ASSESSMENT
---------------
Estimated income bracket: MIDDLE-INCOME
Characteristics: Typical urban traffic with economy and mid-range vehicles.
Good location for:
  - Quick service restaurants
  - Coffee shops
  - Convenience stores
```

## Architecture

```
Video Input → Frame Sampling → Vehicle Detection (YOLOv8) →
Vehicle Tracking (ByteTrack) → Commercial Detection →
Vehicle Classification (CLIP) → Price Lookup → Report Generation
```

## Indonesian Vehicle Context

### Price Tiers (Cars - IDR)
| Tier | Range | Examples |
|------|-------|----------|
| Budget | < 150M | Wuling Air EV, Daihatsu Ayla |
| Economy | 150-300M | Toyota Avanza, Suzuki Ertiga |
| Mid-range | 300-500M | Toyota Innova, Honda HR-V |
| Premium | 500-800M | Toyota Fortuner, Honda CR-V |
| Luxury | > 800M | Toyota Alphard, BMW, Mercedes |

### Price Tiers (Motorcycles - IDR)
| Tier | Range | Examples |
|------|-------|----------|
| Budget | < 18M | Honda Beat, Yamaha Mio |
| Mid | 18-35M | Honda Vario, Yamaha NMAX |
| Premium | > 35M | Honda PCX, Yamaha R15 |

## Configuration

Generate a config file:

```bash
jalanlytics init-config --output myconfig.yaml
```

Use custom config:

```bash
jalanlytics analyze video.mp4 --config myconfig.yaml
```

## Development

### Project Structure

```
jalanlytics/
├── src/
│   ├── main.py              # CLI entry point
│   ├── config.py            # Configuration
│   ├── pipeline.py          # Main orchestrator
│   ├── video/               # Video processing
│   ├── detection/           # Vehicle detection
│   ├── tracking/            # Vehicle tracking
│   ├── classification/      # CLIP classifier
│   ├── data/                # Price database
│   └── report/              # Report generation
├── tests/
├── AGENT_INSTRUCTIONS.md    # Developer guide
├── requirements.txt
└── pyproject.toml
```

### Running Tests

```bash
pytest tests/
```

## License

MIT License

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [OpenAI CLIP](https://github.com/openai/CLIP) for zero-shot classification
- [Supervision](https://github.com/roboflow/supervision) for tracking and visualization

# ── Configuration ─────────────────────────────────────────────────────────────
PYTHON       ?= python
CAMERA_INDEX ?= 0
CLASSES      ?= 3
SAMPLES      ?= 100
DATA_DIR     ?= ./data
THRESHOLD    ?= 0.1
INPUT_WIDTH  ?= 368
INPUT_HEIGHT ?= 368

# Required for 'dataset' and 'infer' targets — no defaults.
PROTO   ?=
WEIGHTS ?=

# ── Helpers ───────────────────────────────────────────────────────────────────
define require
	@[ -n "$($(1))" ] || \
		{ echo "Error: $(1) is required for this target."; \
		  echo "Usage: make $(2) $(1)=/path/to/file"; exit 1; }
endef

# ── Targets ───────────────────────────────────────────────────────────────────
.PHONY: help all collect dataset train infer clean

help:
	@echo "Sign Language Detector — Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all       Run the full training pipeline (collect → dataset → train)"
	@echo "  collect   Capture labeled images from the webcam"
	@echo "  dataset   Extract OpenPose keypoints from images  →  data.pickle"
	@echo "  train     Train the classifier on data.pickle     →  model.p"
	@echo "  infer     Run real-time sign recognition from the webcam"
	@echo "  clean     Remove generated data.pickle and model.p"
	@echo ""
	@echo "Variables (override with VAR=value):"
	@echo "  PROTO          Path to pose_deploy_linevec.prototxt  (required by dataset, infer)"
	@echo "  WEIGHTS        Path to pose_iter_440000.caffemodel    (required by dataset, infer)"
	@echo "  CAMERA_INDEX   Webcam device ID           [$(CAMERA_INDEX)]"
	@echo "  CLASSES        Number of sign classes     [$(CLASSES)]"
	@echo "  SAMPLES        Images captured per class  [$(SAMPLES)]"
	@echo "  DATA_DIR       Image output directory     [$(DATA_DIR)]"
	@echo "  THRESHOLD      Keypoint confidence cutoff [$(THRESHOLD)]"
	@echo ""
	@echo "Example:"
	@echo "  make all PROTO=~/models/pose.prototxt WEIGHTS=~/models/pose.caffemodel"

all: collect dataset train

collect:
	$(PYTHON) collect_imgs.py \
		--camera-index $(CAMERA_INDEX) \
		--classes      $(CLASSES) \
		--samples-per-class $(SAMPLES) \
		--data-dir     $(DATA_DIR)

dataset:
	$(call require,PROTO,dataset)
	$(call require,WEIGHTS,dataset)
	$(PYTHON) create_dataset.py \
		--proto        $(PROTO) \
		--weights      $(WEIGHTS) \
		--threshold    $(THRESHOLD) \
		--input-width  $(INPUT_WIDTH) \
		--input-height $(INPUT_HEIGHT)

train: data.pickle
	$(PYTHON) train_classifier.py

data.pickle:
	@echo "Error: data.pickle not found — run 'make dataset' first."
	@exit 1

infer: model.p
	$(call require,PROTO,infer)
	$(call require,WEIGHTS,infer)
	$(PYTHON) inference_classifier.py \
		--proto        $(PROTO) \
		--weights      $(WEIGHTS) \
		--camera-index $(CAMERA_INDEX) \
		--threshold    $(THRESHOLD) \
		--input-width  $(INPUT_WIDTH) \
		--input-height $(INPUT_HEIGHT)

model.p:
	@echo "Error: model.p not found — run 'make train' first."
	@exit 1

clean:
	rm -f data.pickle model.p

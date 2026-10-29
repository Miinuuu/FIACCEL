PYTHON ?= python3
.DEFAULT_GOAL := help

.PHONY: help check-public host test test-model train_float train_int sim sim-wide sim-waves
help:
	@echo "check-public  Verify protected RTL and approved image checksums"
	@echo "host          Build the U200 XRT host"
	@echo "test          Run host and deployment tests (CPU only)"
	@echo "test-model    Run model and training tests (training environment)"
	@echo "sim           Run the encrypted 32-bit AXI XSim fixture"
	@echo "sim-wide      Run the 256-bit AXI XSim fixture"
	@echo "sim-waves     Run XSim with debug signals and waveforms"
	@echo "train_float / train_int  Train using training/config/"

check-public:
	$(PYTHON) hardware/scripts/check_public.py
host:
	$(MAKE) -C sw host
test: check-public
	$(MAKE) -C sw test PYTHON=$(PYTHON)
test-model:
	$(PYTHON) -m unittest discover -s tests -v
train_float:
	$(PYTHON) -m training.train --config-name train_config_float
train_int:
	$(PYTHON) -m training.train --config-name train_config_int
sim: check-public
	$(PYTHON) hardware/sim/run_xsim.py
sim-wide: check-public
	$(PYTHON) hardware/sim/run_xsim.py --data-width 256 --output build/public_xsim_256
sim-waves: check-public
	$(PYTHON) hardware/sim/run_xsim.py --waves --output build/public_xsim_waves

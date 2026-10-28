train_float:
	python model_train.py --config-name train_config_float
train_int:
	python model_train.py --config-name train_config_int

.PHONY: sim sim-wide check-public
check-public:
	python3 hardware/scripts/check_public.py
sim: check-public
	python3 hardware/scripts/run_xsim.py
sim-wide: check-public
	python3 hardware/scripts/run_xsim.py --data-width 256 --output build/public_xsim_256

.PHONY: sim-waves
sim-waves: check-public
	python3 hardware/scripts/run_xsim.py --waves --output build/public_xsim_waves

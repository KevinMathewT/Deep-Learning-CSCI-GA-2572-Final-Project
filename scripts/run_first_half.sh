#!/bin/bash
# python -m train --config config/areg_jepa_config_1_1_1_1.yaml
python -m train --config config/areg_jepa_config_1_1_1_25.yaml
python -m train --config config/areg_jepa_config_1_1_25_1.yaml
python -m train --config config/areg_jepa_config_1_1_25_25.yaml
python -m train --config config/areg_jepa_config_1_25_1_1.yaml
python -m train --config config/areg_jepa_config_1_25_1_25.yaml
python -m train --config config/areg_jepa_config_1_25_25_1.yaml
python -m train --config config/areg_jepa_config_1_25_25_25.yaml

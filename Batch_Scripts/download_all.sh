#!/bin/bash
mkdir -p /cluster/home/pdamota/openml_cache
echo 'Starting Bulk Download for all Suites...'

# --- Processing Suite 334 ---
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44156
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44156/dataset.arff 'https://openml.org/data/v1/download/22103281/electricity.arff'
echo 'Downloaded Dataset 44156 (electricity)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44157
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44157/dataset.arff 'https://openml.org/data/v1/download/22103282/eye_movements.arff'
echo 'Downloaded Dataset 44157 (eye_movements)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44159
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44159/dataset.arff 'https://openml.org/data/v1/download/22103284/covertype.arff'
echo 'Downloaded Dataset 44159 (covertype)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45035
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45035/dataset.arff 'https://openml.org/data/v1/download/22111925/albert.arff'
echo 'Downloaded Dataset 45035 (albert)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45036
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45036/dataset.arff 'https://openml.org/data/v1/download/22111926/default-of-credit-card-clients.arff'
echo 'Downloaded Dataset 45036 (default-of-credit-card-clients)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45038
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45038/dataset.arff 'https://openml.org/data/v1/download/22111928/road-safety.arff'
echo 'Downloaded Dataset 45038 (road-safety)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45039
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45039/dataset.arff 'https://openml.org/data/v1/download/22111929/compas-two-years.arff'
echo 'Downloaded Dataset 45039 (compas-two-years)'

# --- Processing Suite 335 ---
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44055
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44055/dataset.arff 'https://openml.org/data/v1/download/22103151/analcatdata_supreme.arff'
echo 'Downloaded Dataset 44055 (analcatdata_supreme)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44056
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44056/dataset.arff 'https://openml.org/data/v1/download/22103152/visualizing_soil.arff'
echo 'Downloaded Dataset 44056 (visualizing_soil)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44059
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44059/dataset.arff 'https://openml.org/data/v1/download/22103155/diamonds.arff'
echo 'Downloaded Dataset 44059 (diamonds)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44061
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44061/dataset.arff 'https://openml.org/data/v1/download/22103157/Mercedes_Benz_Greener_Manufacturing.arff'
echo 'Downloaded Dataset 44061 (Mercedes_Benz_Greener_Manufacturing)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44062
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44062/dataset.arff 'https://openml.org/data/v1/download/22103158/Brazilian_houses.arff'
echo 'Downloaded Dataset 44062 (Brazilian_houses)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44063
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44063/dataset.arff 'https://openml.org/data/v1/download/22103159/Bike_Sharing_Demand.arff'
echo 'Downloaded Dataset 44063 (Bike_Sharing_Demand)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44065
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44065/dataset.arff 'https://openml.org/data/v1/download/22103161/nyc-taxi-green-dec-2016.arff'
echo 'Downloaded Dataset 44065 (nyc-taxi-green-dec-2016)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44066
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44066/dataset.arff 'https://openml.org/data/v1/download/22103162/house_sales.arff'
echo 'Downloaded Dataset 44066 (house_sales)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44068
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44068/dataset.arff 'https://openml.org/data/v1/download/22103164/particulate-matter-ukair-2017.arff'
echo 'Downloaded Dataset 44068 (particulate-matter-ukair-2017)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44069
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44069/dataset.arff 'https://openml.org/data/v1/download/22103165/SGEMM_GPU_kernel_performance.arff'
echo 'Downloaded Dataset 44069 (SGEMM_GPU_kernel_performance)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45041
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45041/dataset.arff 'https://openml.org/data/v1/download/22111939/topo_2_1.arff'
echo 'Downloaded Dataset 45041 (topo_2_1)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45042
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45042/dataset.arff 'https://openml.org/data/v1/download/22111940/abalone.arff'
echo 'Downloaded Dataset 45042 (abalone)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45043
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45043/dataset.arff 'https://openml.org/data/v1/download/22111941/seattlecrime6.arff'
echo 'Downloaded Dataset 45043 (seattlecrime6)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45045
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45045/dataset.arff 'https://openml.org/data/v1/download/22111943/delays_zurich_transport.arff'
echo 'Downloaded Dataset 45045 (delays_zurich_transport)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45046
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45046/dataset.arff 'https://openml.org/data/v1/download/22111944/Allstate_Claims_Severity.arff'
echo 'Downloaded Dataset 45046 (Allstate_Claims_Severity)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45047
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45047/dataset.arff 'https://openml.org/data/v1/download/22111945/Airlines_DepDelay_1M.arff'
echo 'Downloaded Dataset 45047 (Airlines_DepDelay_1M)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45048
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45048/dataset.arff 'https://openml.org/data/v1/download/22111946/medical_charges.arff'
echo 'Downloaded Dataset 45048 (medical_charges)'

# --- Processing Suite 336 ---
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44132
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44132/dataset.arff 'https://openml.org/data/v1/download/22103257/cpu_act.arff'
echo 'Downloaded Dataset 44132 (cpu_act)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44133
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44133/dataset.arff 'https://openml.org/data/v1/download/22103258/pol.arff'
echo 'Downloaded Dataset 44133 (pol)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44134
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44134/dataset.arff 'https://openml.org/data/v1/download/22103259/elevators.arff'
echo 'Downloaded Dataset 44134 (elevators)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44136
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44136/dataset.arff 'https://openml.org/data/v1/download/22103261/wine_quality.arff'
echo 'Downloaded Dataset 44136 (wine_quality)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44137
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44137/dataset.arff 'https://openml.org/data/v1/download/22103262/Ailerons.arff'
echo 'Downloaded Dataset 44137 (Ailerons)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44138
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44138/dataset.arff 'https://openml.org/data/v1/download/22103263/houses.arff'
echo 'Downloaded Dataset 44138 (houses)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44139
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44139/dataset.arff 'https://openml.org/data/v1/download/22103264/house_16H.arff'
echo 'Downloaded Dataset 44139 (house_16H)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44140
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44140/dataset.arff 'https://openml.org/data/v1/download/22103265/diamonds.arff'
echo 'Downloaded Dataset 44140 (diamonds)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44141
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44141/dataset.arff 'https://openml.org/data/v1/download/22103266/Brazilian_houses.arff'
echo 'Downloaded Dataset 44141 (Brazilian_houses)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44142
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44142/dataset.arff 'https://openml.org/data/v1/download/22103267/Bike_Sharing_Demand.arff'
echo 'Downloaded Dataset 44142 (Bike_Sharing_Demand)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44143
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44143/dataset.arff 'https://openml.org/data/v1/download/22103268/nyc-taxi-green-dec-2016.arff'
echo 'Downloaded Dataset 44143 (nyc-taxi-green-dec-2016)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44144
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44144/dataset.arff 'https://openml.org/data/v1/download/22103269/house_sales.arff'
echo 'Downloaded Dataset 44144 (house_sales)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44145
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44145/dataset.arff 'https://openml.org/data/v1/download/22103270/sulfur.arff'
echo 'Downloaded Dataset 44145 (sulfur)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44146
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44146/dataset.arff 'https://openml.org/data/v1/download/22103271/medical_charges.arff'
echo 'Downloaded Dataset 44146 (medical_charges)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44147
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44147/dataset.arff 'https://openml.org/data/v1/download/22103272/MiamiHousing2016.arff'
echo 'Downloaded Dataset 44147 (MiamiHousing2016)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44148
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44148/dataset.arff 'https://openml.org/data/v1/download/22103273/superconduct.arff'
echo 'Downloaded Dataset 44148 (superconduct)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45032
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45032/dataset.arff 'https://openml.org/data/v1/download/22111920/yprop_4_1.arff'
echo 'Downloaded Dataset 45032 (yprop_4_1)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45033
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45033/dataset.arff 'https://openml.org/data/v1/download/22111921/abalone.arff'
echo 'Downloaded Dataset 45033 (abalone)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45034
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45034/dataset.arff 'https://openml.org/data/v1/download/22111922/delays_zurich_transport.arff'
echo 'Downloaded Dataset 45034 (delays_zurich_transport)'

# --- Processing Suite 337 ---
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44089
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44089/dataset.arff 'https://openml.org/data/v1/download/22103185/credit.arff'
echo 'Downloaded Dataset 44089 (credit)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44120
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44120/dataset.arff 'https://openml.org/data/v1/download/22103245/electricity.arff'
echo 'Downloaded Dataset 44120 (electricity)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44121
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44121/dataset.arff 'https://openml.org/data/v1/download/22103246/covertype.arff'
echo 'Downloaded Dataset 44121 (covertype)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44122
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44122/dataset.arff 'https://openml.org/data/v1/download/22103247/pol.arff'
echo 'Downloaded Dataset 44122 (pol)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44123
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44123/dataset.arff 'https://openml.org/data/v1/download/22103248/house_16H.arff'
echo 'Downloaded Dataset 44123 (house_16H)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44125
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44125/dataset.arff 'https://openml.org/data/v1/download/22103250/MagicTelescope.arff'
echo 'Downloaded Dataset 44125 (MagicTelescope)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44126
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44126/dataset.arff 'https://openml.org/data/v1/download/22103251/bank-marketing.arff'
echo 'Downloaded Dataset 44126 (bank-marketing)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44128
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44128/dataset.arff 'https://openml.org/data/v1/download/22103253/MiniBooNE.arff'
echo 'Downloaded Dataset 44128 (MiniBooNE)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44129
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44129/dataset.arff 'https://openml.org/data/v1/download/22103254/Higgs.arff'
echo 'Downloaded Dataset 44129 (Higgs)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44130
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/44130/dataset.arff 'https://openml.org/data/v1/download/22103255/eye_movements.arff'
echo 'Downloaded Dataset 44130 (eye_movements)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45022
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45022/dataset.arff 'https://openml.org/data/v1/download/22111908/Diabetes130US.arff'
echo 'Downloaded Dataset 45022 (Diabetes130US)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45021
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45021/dataset.arff 'https://openml.org/data/v1/download/22111907/jannis.arff'
echo 'Downloaded Dataset 45021 (jannis)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45020
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45020/dataset.arff 'https://openml.org/data/v1/download/22111906/default-of-credit-card-clients.arff'
echo 'Downloaded Dataset 45020 (default-of-credit-card-clients)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45019
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45019/dataset.arff 'https://openml.org/data/v1/download/22111905/Bioresponse.arff'
echo 'Downloaded Dataset 45019 (Bioresponse)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45028
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45028/dataset.arff 'https://openml.org/data/v1/download/22111914/california.arff'
echo 'Downloaded Dataset 45028 (california)'
mkdir -p /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45026
wget -nc -q --show-progress -O /cluster/home/pdamota/openml_cache/org/openml/www/datasets/45026/dataset.arff 'https://openml.org/data/v1/download/22111912/heloc.arff'
echo 'Downloaded Dataset 45026 (heloc)'

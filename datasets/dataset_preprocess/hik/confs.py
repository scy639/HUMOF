STEP_train=17
STEP_test=41

# others filter
ENABLE_filterB:bool = 1
THRES_filterB = 8.0
THRES_filterB_train = 9.9

# primary-person related filter
PRIMARY_FILTER_C:bool = 1
PRIMARY_FILTER_C2:bool = 1  # only valid when PRIMARY_FILTER_C (PRIMARY_FILTER_C 's sub conf
THRES__primary_filterC_test = 0.4
THRES__primary_filterC_train = 0.6
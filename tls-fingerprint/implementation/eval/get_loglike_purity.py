import os
import sys
import numpy as np
import json as js

sys.path.append('/mounted-data/code/')

from baum_welch import phmm, learning
from data_conversion import data_aggregation
from data_conversion import constants
from hmm_tf import fingerprint_hmm

def acquire_log_like_for_purity():

    browser = constants.BROWSER_MOZILLA
    companys = data_aggregation.get_services()
    company = 'youtube'
    index = companys.index(company)

    company_l = []
    rest_l = []

    json_files = data_aggregation.get_traces_test(browser)

    json_files = json_files[25*index:25*(index + 1)]

    for json_file in json_files:

        js_data = data_aggregation.load_prediction(os.path.join(constants.classification_path, json_file + '_clas.json'))

        likelihoods = js_data['likelihoods']
        actual = likelihoods[company]
        company_l.append(actual)

        likelihoods.pop(company)

        for likelihood in likelihoods.values():
            rest_l.append(likelihood)

    output = dict()
    output[company] = company_l
    output['rest'] = rest_l

    output_file = os.path.join(constants.result_path, 'purity', company + '_' + browser +'_thres.json')

    with open(output_file, 'w', encoding = 'utf-8') as f:
        js.dump(output, f, ensure_ascii = False, indent = 4)


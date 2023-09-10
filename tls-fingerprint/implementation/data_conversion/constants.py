
# Literals

# Browser:
BROWSER_MOZILLA = 'Mozilla'
BROWSER_CHROME = 'Chromium'
BROWSER_WGET = 'Wget'
BROWSER_NOT_WGET = 'not_wget'

# Flows:

FLOW_MAIN = 'main_flow'
FLOW_CO = 'co_flows'
SEQ_TYPE_FRAME = 'frame'
SEQ_TYPE_RECORD = 'record'

# Binning:

BINNING_GEOM = 'geometric'
BINNING_FREQ = 'frequency'
BINNING_NONE = 'None'
BINNING_SINGLE = 'singlebin'
BINNING_EQ_WIDTH = 'equal_width'
BINNING_NONE = None

#=================================================================================================================

# Paths:

data_dir = '/mounted-data/data'
trace_path = '/mounted-data/data/traces/pcapng'
model_path = '/mounted-data/data/models'
classification_path = '/mounted-data/data/classification'
result_path = '/mounted-data/data/results'
tf_log_dir = '/mounted-data/data/logs'
# data_dir = '/opt/project/data'
# trace_path = '/opt/project/data/traces/pcapng'
# model_path = '/opt/project/data/models'
# classification_path = '/opt/project/data/classification'
# result_path = '/opt/project/data/results'
# tf_log_dir = '/opt/project/data/logs'

#=================================================================================================================

# States:

tls_states_mc = dict()
tls_states_mc['Change Cipher Spec'] = '20'
tls_states_mc['Encrypted Alert'] = '21'
tls_states_mc['Encrypted Handshake Message'] = '22'
tls_states_mc['Client Hello'] = '22:1'
tls_states_mc['Server Hello'] = '22:2'
tls_states_mc['New Session Ticket'] = '22:4'
tls_states_mc['Certificate'] = '22:11'
tls_states_mc['Server Key Exchange'] = '22:12'
tls_states_mc['Server Hello Done'] = '22:14'
tls_states_mc['Client Key Exchange'] = '22:16'
tls_states_mc['Certificate Status'] = '22:22'
tls_states_mc['Application Data'] = '23'
tls_states_mc['Continuation Data'] = '23'


#=================================================================================================================

# Database

SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://root:FKmk2kRzWFFdX8@pk-swc01.forschung.lkn.ei.tum.de:3306/gatherer_upscaled'

#=================================================================================================================

# HMM: (Tensorflow)

client_states = 5
server_states = 9
num_emissions = None

tls_states_hmm_c = dict()
tls_states_hmm_c['Client Hello'] = 0
tls_states_hmm_c['Client Key Exchange'] = 1
tls_states_hmm_c['Change Cipher Spec'] = 2
tls_states_hmm_c['Encrypted Handshake Message'] = 3
tls_states_hmm_c['Application Data'] = 4
tls_states_hmm_c['Continuation Data'] = 4

tls_states_hmm_s = dict()
tls_states_hmm_s['Server Hello'] = 5
tls_states_hmm_s['Server Key Exchange'] = 6
tls_states_hmm_s['Change Cipher Spec'] = 7
tls_states_hmm_s['Encrypted Handshake Message'] = 8
tls_states_hmm_s['Application Data'] = 9
tls_states_hmm_s['Continuation Data'] = 9
tls_states_hmm_s['Certificate'] = 10
tls_states_hmm_s['Certificate Status'] = 11
tls_states_hmm_s['New Session Ticket'] = 12
tls_states_hmm_s['Server Hello Done'] = 13


default_state = (client_states + server_states)

#=================================================================================================================

applications = ["amazon", "facebook", "google", "google_drive", "google_maps", "wikipedia", "youtube"]
applications_short = ["az", "fb", "gs", "gd", "gm", "wp", "yt", "ukn"]

opt_bins_no = {2: 79, 3: 79, 4: 97, 5: 97, 6: 96, 7: 76, 8: 87, 9: 78, 10: 78, 11: 100, 12: 76, 13: 76, 14: 76, 15: 95, 16: 76, 17: 76, 18: 76, 19: 76, 20: 76, 21: 76, 22: 76, 23: 76, 24: 76, 25: 76, 26: 76, 27: 76, 28: 76, 29: 76, 30: 76}
opt_bins_co = {2: 79, 3: 79, 4: 97, 5: 95, 6: 90, 7: 90, 8: 90, 9: 72, 10: 79, 11: 75, 12: 99, 13: 74, 14: 95, 15: 97, 16: 77, 17: 88, 18: 82, 19: 88, 20: 70, 21: 70, 22: 70, 23: 70, 24: 70, 25: 70, 26: 70, 27: 70, 28: 70, 29: 70, 30: 70}


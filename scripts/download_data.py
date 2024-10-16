import sys

from datetime import datetime, timedelta
from doe_dap_dl import DAP

if len(sys.argv) == 1:
    start_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d000000") 
    end_date = (datetime.now()).strftime("%Y%m%d000000")
elif len(sys.argv) == 2:
    start_date = '%s000000' % sys.argv[1]
    end_date = (datetime.strptime(sys.argv[1], '%Y%m%d') + timedelta(days=1)).strftime("%Y%m%d000000")
elif len(sys.argv) == 3:
    start_date = '%s000000' % sys.argv[1]
    end_date = '%s000000' % sys.argv[2]

dest = '/lcrc/group/earthscience/rjackson/wfip3/barg/mrr/'
a2e = DAP('a2e.energy.gov', confirm_downloads=False)
a2e.setup_basic_auth('rjackson@anl.gov', 'Kur@do43c')
filt = {
    'Dataset': 'wfip3/caco.lidar.z02.00',
    'date_time': {
        'between': [start_date, end_date]
    },
    'file_type': 'hpl'
}
files = a2e.search(filt)
a2e.download_files(files, path=dest, replace=False)

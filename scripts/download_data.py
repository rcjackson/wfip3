
from doe_dap_dl import DAP

dest = '/lcrc/group/earthscience/rjackson/wfip3/caco/lidar'
a2e = DAP('a2e.energy.gov')
a2e.setup_basic_auth('rjackson@anl.gov', 'Kur@do43c')
filt = {
    'Dataset': 'wfip3/caco.lidar.z02.00',
    'date_time': {
        'between': ['20240415000000', '20240418000000']
    },
    'file_type': 'hpl'
}
files = a2e.search(filt)
a2e.download_files(files, path=dest, replace=False)

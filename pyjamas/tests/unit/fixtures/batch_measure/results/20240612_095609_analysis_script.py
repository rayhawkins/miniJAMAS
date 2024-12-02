parameters = {'folders': ['/Users/rodrigo/src/pyjamas_dev/pyjamas/tests/unit/fixtures/batch_measure/water_1', '/Users/rodrigo/src/pyjamas_dev/pyjamas/tests/unit/fixtures/batch_measure/water_2', '/Users/rodrigo/src/pyjamas_dev/pyjamas/tests/unit/fixtures/batch_measure/juice'], 'analyze_flag': True, 'analysis_filename_appendix': '_analysis', 'analysis_extension': '.csv', 'save_results': True, 'script_filename_appendix': '_analysis_script.ipynb', 'results_folder': '/Users/rodrigo/src/pyjamas_dev/pyjamas/tests/unit/fixtures/batch_measure/results', 'intensity_flag': True, 'image_extension': '.tif', 'normalize_intensity_flag': 0, 't_res': 30.0, 'xy_res': 0.178, 'index_time_zero': 2, 'plot_flag': False, 'names': ['water 1', 'water 2', 'juice'], 'err_style_value': 'band', 'plot_style_value': 'box', 'brush_sz': 3}
import sys
sys.path.extend(['/Users/rodrigo/src/pyjamas_dev'])
from pyjamas.pjscore import PyJAMAS
a = PyJAMAS()
a.batch.cbMeasureBatch(parameters)

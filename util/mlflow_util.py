import mlflow
import os
import numpy as np
import urllib.parse

def get_prev_run(function, params, tags=None, git_commit=None):
    query = 'attributes.status = "FINISHED"'
    query += ' and tags."function" = "{}"'.format(function)
    for key, val in params.items():
        query += ' and '
        query += 'params.{} = "{}"'.format(key, val) 
    if tags:
        for key, val in tags.items():
            query += ' and '
            query += 'tags."{}" = "{}"'.format(key, val)
    runs = mlflow.search_runs(filter_string=query)
    if runs.empty:
        return None
    else:
        # TODO should check git_commit
        return mlflow.tracking.MlflowClient().get_run(
            runs.iloc[0].loc['run_id'])

def load_uri(uri):
    url_data = urllib.parse.urlparse(uri)
    path = urllib.parse.unquote(url_data.path)
    return np.load(path)
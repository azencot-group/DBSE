import neptune.new as neptune

def init_neptune(opt, neptune_project_name, neptune_api_token, neptune_api_tags):
    run = neptune.init_run(
        project= neptune_project_name,
        api_token= neptune_api_token,
    )  # Your credentials
    run['config/hyperparameters'] = vars(opt)
    run['config/name'] = neptune_api_token
    run["sys/tags"].add(neptune_api_tags)
    return run

def log_to_neptune(run, neptune_log_dic):
    for key, value in neptune_log_dic.items():
        run[key].log(value)

def upload_image_to_neptune(run, neptune_log_dic):
    for key, value in neptune_log_dic.items():
        run[key].upload(value)
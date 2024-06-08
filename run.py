'''
Pipeline to run the hard arena.
'''
import os
import yaml
import argparse
import time
from azureml.core import Run

def get_endpoints_key_map(endpoints, is_aml_run):
    endpoint_key_map = {}

    if is_aml_run == "True":
        # for aml run, we get AOAI key from keyvault of the AMl workspace.
        run = Run.get_context()
        ws = run.experiment.workspace
        keyvault = ws.get_default_keyvault()

        for endpoint in endpoints:
            endpt_name = endpoint["name"]
            endpt_key_name = endpoint["name"] + "-aoai-key"
            endpt_key = keyvault.get_secret(endpt_key_name)

            endpoint_key_map[endpt_name] = endpt_key
    else:    
        # for debug local run, add mapping here
        pass

    return endpoint_key_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id", type=str, default="Phi-3-mini-4k-instruct"
    )
    parser.add_argument(
        "--model_name", type=str, help="model name corresponds to vllm server"
    )
    parser.add_argument(
        "--is_aml_run", type=str, default="True", help="if it is an AML run"
    )
    parser.add_argument(
        "--input_dir", type=str, default="None", help="input dir for AML run"
    )
    parser.add_argument(
        "--output_dir", type=str, default="None", help="output dir for AML run"
    )
    parser.add_argument(
        "--port", type=str, default="8008", help="port for hosting vllm"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode"
    )
    parser.add_argument(
        "--mode", type=str, default="all", help="mode to run, all or gen_answer or gen_judgment",
        choices=["all", "gen_answer", "gen_judgment"]
    )
    args = parser.parse_args()

    endpoints = [{"name": "gpt-4-1106-preview-4eval"}]
    endpoint_key_map = get_endpoints_key_map(endpoints, args.is_aml_run)

    # read the api_config file
    file_path = os.path.join(os.path.dirname(__file__), 'config/api_config.yaml')
    with open(file_path, 'r') as file:
        api_config = yaml.safe_load(file)

    # add judge model api
    if args.is_aml_run == "True":
        api_key = endpoint_key_map["gpt-4-1106-preview-4eval"]
    else:
        api_key = os.environ['AOAI_API_KEY']
    
    judge_model_name = 'tscience-uks-gpt4-1106'
    add_dict = {judge_model_name: 
                    {'model_name': judge_model_name, 
                    'endpoints': [{
                                    'api_base': 'https://aims-oai-research-inference-uks.openai.azure.com/', 
                                    'api_key': api_key, 
                                    'api_version': '2024-02-01'
                                }],
                    'api_type': 'azure', 
                    'parallel': 8,
                    }
                }
    api_config.update(add_dict)

    # add test model api
    model_id = args.model_id
    model_name = args.model_name

    add_dict = {model_id: 
                    {'model_name': model_name, 
                    'endpoints': [{'api_base': f'http://localhost:{args.port}/v1', 'api_key': 'token-abc123'}], 
                    'api_type': 'openai', 
                    'parallel': 16
                    }
                }

    api_config.update(add_dict)
    print(api_config)
    file_path = os.path.join(os.path.dirname(__file__), 'config/api_config_test.yaml')
    with open(file_path, 'w') as file:
        yaml.safe_dump(api_config, file, default_flow_style=False)


    # read the gen_answer_config file
    file_path = os.path.join(os.path.dirname(__file__), 'config/gen_answer_config.yaml')
    with open(file_path, 'r') as file:
        gen_answer_config = yaml.safe_load(file)

    gen_answer_config['model_list'] = [model_id]
    gen_answer_config['max_tokens'] = 2048 # reduce this from 4096 to 2048 since phi model has a limit of 4k tokens

    print(gen_answer_config)
    file_path = os.path.join(os.path.dirname(__file__), 'config/gen_answer_config_test.yaml')
    with open(file_path, 'w') as file:
        yaml.safe_dump(gen_answer_config, file, default_flow_style=False)


    # read the judge_config file
    file_path = os.path.join(os.path.dirname(__file__), 'config/judge_config.yaml')
    with open(file_path, 'r') as file:
        judge_config = yaml.safe_load(file)

    judge_config['judge_model'] = judge_model_name
    judge_config['model_list'] = [model_id]

    print(judge_config)
    file_path = os.path.join(os.path.dirname(__file__), 'config/judge_config_test.yaml')
    with open(file_path, 'w') as file:
        yaml.safe_dump(judge_config, file, default_flow_style=False)

    if args.mode in ["gen_answer", "all"]:
        # start the model vllm hosting
        os.system(f'nohup python -m vllm.entrypoints.openai.api_server --model {model_name} --dtype auto --api-key token-abc123 --port {args.port} > server_output.log 2>&1 &')

        # wait for the server to start
        time.sleep(30)

        # run answer generation
        os.system('python gen_answer.py --setting-file config/gen_answer_config_test.yaml --endpoint-file config/api_config_test.yaml' + ' --debug' if args.debug else '')

        # stop the model vllm hosting
        os.system('pkill -f vllm.entrypoints.openai.api_server')

        # wait for the server to stop
        time.sleep(10)

        # copy results to output dir
        if args.output_dir != "None":
            os.system(f'cp -r data/arena-hard-v0.1 {args.output_dir}')

    if args.mode in ["gen_judgment", "all"]:
        if args.input_dir != "None":
            os.system(f'cp -r {args.input_dir}/arena-hard-v0.1 data/')
        
        # run judgement generation
        os.system('python gen_judgment.py --setting-file config/judge_config_test.yaml --endpoint-file config/api_config_test.yaml')

        # show results
        os.system(f'python show_result.py --judge-name {judge_model_name}')

        # copy results to output dir
        if args.output_dir != "None":
            os.system(f'cp -r data/arena-hard-v0.1 {args.output_dir}')
from rkllm.api import RKLLM
from huggingface_hub import login, whoami, snapshot_download, auth_check, ModelCard, HfApi
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
from pathlib import Path
import inquirer
import shutil
import os
import gc


def parse_context_length(value):
    """Parses context length, validates, aligns, and determines the best suffix."""
    MAX_CONTEXT = 16384
    num = 0
    try:
        if isinstance(value, str) and value.lower().endswith('k'):
            # Allow float for values like '1.5k'
            num = int(float(value[:-1]) * 1024)
        else:
            num = int(value)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid context length format: {value}")

    # Check if the initial value exceeds the maximum
    if num > MAX_CONTEXT:
        raise ValueError(
            f"Context length {num} exceeds the maximum of {MAX_CONTEXT}.")

    # Align to the nearest multiple of 32
    aligned_num = (num + 31) // 32 * 32
    if aligned_num != num:
        print(f"Warning: Context length {num} was aligned to {aligned_num}.")

    # Check if the aligned value now exceeds the maximum
    if aligned_num > MAX_CONTEXT:
        raise ValueError(
            f"Aligned context length {aligned_num} exceeds the maximum of {MAX_CONTEXT}.")

    # Determine the best short name for the suffix
    if aligned_num % 1024 == 0:
        short_name = f"{aligned_num // 1024}k"
    else:
        short_name = str(aligned_num)

    return aligned_num, short_name


class RKLLMRemotePipeline:
    def __init__(self, model_id="", lora_id="",
                 platform="rk3588", library_type="HF"):
        """
        Initialize primary values for pipeline class.
        """
        self.model_id = model_id
        self.lora_id = lora_id
        self.platform = platform
        self.library_type = library_type
        self.model_name = self.model_id.split("/", 1)[1]
        self.model_dir = f"./models/{self.model_name}/"
        # Set the requested version here
        self.rkllm_version = "1.2.1"

        if self.platform == "rk3588":
            self.npu_cores = 3
        elif self.platform == "rk3576":
            self.npu_cores = 2

        self.dataset = None
        self.qparams = None
        self.device = "cuda"
        self.rkllm = None

    @staticmethod
    def mkpath(path):
        """
        HuggingFace Hub will just fail if the local_dir you are downloading to does not exist
        RKLLM will also fail to export if the directory does not already exist.
        """
        try:
            if not os.path.exists(path):
                os.makedirs(path)
                print(f"mkdir'd {path}")
            else:
                print(f"Path {path} already exists! Great job!")
        except RuntimeError as e:
            print(
                f"Can't create paths for importing and exporting model.\n{e}")

    @staticmethod
    def cleanup_models(path=Path("./models")):
        if os.path.exists(path):
            print("Cleaning up model directory...")
            shutil.rmtree(path)

    def initialize_and_load(self):
        '''
        Handles the one-time setup: path creation, model download, and loading into memory.
        '''
        print(f"Checking if {self.model_dir} exists...")
        self.mkpath(self.model_dir)

        print(
            f"Loading base model {self.model_id} from HuggingFace and downloading to {self.model_dir}")
        self.modelpath = snapshot_download(
            repo_id=self.model_id, local_dir=self.model_dir)

        self.lorapath = None
        if self.lora_id:
            print(f"Downloading LoRA: {self.lora_id}")
            self.lora_name = self.lora_id.split("/", 1)[1]
            self.lora_dir = f"./models/{self.lora_name}/"
            self.mkpath(self.lora_dir)
            try:
                self.lorapath = snapshot_download(
                    repo_id=self.lora_id, local_dir=self.lora_dir)
            except Exception as e:
                print(f"Downloading LoRA failed: {e}. Omitting from export.")
                self.lorapath = None

        print("Initializing RKLLM class...")
        self.rkllm = RKLLM()

        print("Loading model into memory... (This happens only once per model)")
        if self.library_type == "HF":
            status = self.rkllm.load_huggingface(
                model=self.modelpath, model_lora=self.lorapath, device=self.device)
        elif self.library_type == "GGUF":
            status = self.rkllm.load_gguf(model=self.modelpath)
        else:
            raise RuntimeError("Model must be of type HF or GGUF.")

        if status != 0:
            raise RuntimeError(f"Failed to load model: {status}")
        print(f"{self.model_name} loaded successfully!")

    def build_and_export(self, qtype="w8a8",
                         hybrid_rate="0.0", optimization=1, max_context=4096, context_suffix="4k"):
        '''
        Builds and exports the model for a specific configuration.
        Returns the export path and name for use in other functions (like uploading).
        '''
        name_suffix = f"{self.platform}-{qtype}-opt-{optimization}-hybrid-ratio-{hybrid_rate}-{context_suffix}"
        if self.lora_id and hasattr(self, 'lora_name'):
            export_name = f"{self.model_name}-{self.lora_name}-{name_suffix}"
            export_path = f"./models/{self.model_name}-{self.lora_name}-{self.platform}/"
        else:
            export_name = f"{self.model_name}-{name_suffix}"
            export_path = f"./models/{self.model_name}-{self.platform}/"

        self.mkpath(export_path)

        print(
            f"\nBuilding {self.model_name} with qtype={qtype}, opt={optimization}, hybrid_rate={hybrid_rate}, context={max_context}")
        status = self.rkllm.build(
            optimization_level=optimization,
            quantized_dtype=qtype,
            target_platform=self.platform,
            num_npu_core=self.npu_cores,
            extra_qparams=self.qparams,
            dataset=self.dataset,
            max_context=max_context
        )

        if status != 0:
            print(
                f"Failed to build model for config: qtype={qtype}, opt={optimization}")
            return None, None

        print(f"{self.model_name} built successfully!")

        export_file = f"{export_path}{export_name}.rkllm"
        status = self.rkllm.export_rkllm(export_file)

        if status != 0:
            print(f"Failed to export model to {export_file}")
            return None, None

        print(f"{self.model_name} exported successfully to {export_file}!")
        return export_path, export_name


class HubHelpers:
    def __init__(self, platform, model_id, lora_id, rkllm_version):
        self.model_id = model_id
        self.lora_id = lora_id
        self.platform = platform
        self.rkllm_version = rkllm_version
        self.home_dir = os.environ['HOME']
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

    @staticmethod
    def repo_check(model):
        try:
            auth_check(model)
        except GatedRepoError:
            print(f"{model} is a gated repo.\nYou do not have permission to access it.\n \
                  Please authenticate.\n")
        except RepositoryNotFoundError:
            print(f"{model} not found.")
        else:
            print(f"Model repo {model} has been validated!")
            return True

    def login_to_hf(self):
        self.token_path = f"{self.home_dir}/.cache/huggingface/token"
        if os.path.exists(self.token_path):
            with open(self.token_path, "r") as token_file:
                self.hf_token = token_file.read()
        else:
            hf_input = [
                inquirer.Text(
                    "token",
                    message="Please enter your Hugging Face token",
                    default="")]
            self.hf_token = inquirer.prompt(hf_input)["token"]
        try:
            login(token=self.hf_token)
            self.hf_username = whoami(self.hf_token)["name"]
            print(f"Logged into HuggingFace as {self.hf_username}!")
        except Exception as e:
            print(
                f"Login failed: {e}\nGated models will be inaccessible, and you will not be able to upload to HuggingFace.")
            self.hf_username = None

    def build_card(self, export_path, successful_conversions):
        self.model_name = self.model_id.split("/", 1)[1]
        card_in = ModelCard.load(self.model_id)
        card_out = os.path.join(export_path, "README.md")

        lora_text = f'This model has been optimized with the following LoRA: `{self.lora_id}`\n\n' if self.lora_id else ''

        files_table = "| Quantization | Optimization | Hybrid Ratio | Context | Filename |\n"
        files_table += "|---|---|---|---|---|\n"
        for conv in successful_conversions:
            files_table += f"| `{conv['qtype']}` | `{conv['opt']}` | `{conv['hybrid_rate']}` | `{conv['context']}` | `{conv['filename']}` |\n"

        template = (
            f'---\n{card_in.data.to_yaml()}\nbase_model: {self.model_id}\ntags:\n- rkllm\n---\n'
            f'# {self.model_name} for {self.platform.upper()} (RKLLM {self.rkllm_version})\n\n'
            f'This repository contains versions of `{self.model_id}` converted to run on the {self.platform.upper()} NPU.\n\n'
            f'{lora_text}'
            f'Compatible with RKLLM version: **{self.rkllm_version}**\n\n'
            '## Available Models\n\n'
            f'{files_table}\n\n'
            '## Model Selection\n\n'
            'Ungrouped (non- _gxxx) models are the fastest. Smaller group sizes are slower but may yield better accuracy.\n\n'
            'Enabling quantization precision optimization results in less performance but higher accuracy.\n\n'
            'Hybrid models have a certain ratio of weights that are ungrouped or grouped, depending on the default for the qtype. Reduces effect of ungrouping or grouping type.\n\n'
            '## Useful links:\n'
            '[RKLLM GitHub](https://github.com/airockchip/rknn-llm)\n\n'
            'Converted using ez-rkllm-converter\n\n'
            f'# Original Model Card for base model, {self.model_name}, below:\n\n'
            f'{card_in.text}'
        )
        try:
            with open(card_out, 'w', encoding='utf-8') as f:
                f.write(template)
            print(f"Model card successfully exported to {card_out}!")
        except Exception as e:
            print(f"Error saving model card: {e}")

    def upload_to_repo(self, repo_name, import_path, export_path):
        if not self.hf_username:
            print("Not logged in. Skipping upload.")
            return

        self.hf_api = HfApi(token=self.hf_token)
        repo_id = f"{self.hf_username}/{repo_name}"

        print(f"Creating repo {repo_id} if it does not already exist")
        try:
            repo_url = self.hf_api.create_repo(exist_ok=True, repo_id=repo_id)
            print(f"Repo URL: {repo_url}")
        except Exception as e:
            print(f"Failed to create repo for {repo_name}: {e}")
            return

        print("Searching for and copying LICENSE file...")
        found_license = False
        for filename in os.listdir(import_path):
            if 'license' in filename.lower():
                shutil.copy2(os.path.join(import_path, filename), export_path)
                print(f"Copied '{filename}' to export directory.")
                found_license = True
                break
        if not found_license:
            print("Warning: No LICENSE file found in original repo.")

        print(f"Uploading contents of {export_path} to {repo_id}")
        self.hf_api.upload_folder(
            repo_id=repo_id,
            folder_path=export_path
        )


if __name__ == "__main__":
    # --- Configuration ---
    model_ids = ["Qwen/Qwen3-4B-Thinking-2507"]
    platform = "rk3588"
    # Refer to the SDK for supported qtypes for your platform
    qtypes = ["w8a8_g128", "w8a8"]
    # Use float strings
    hybrid_rates = ["0.0", "0.2", "0.4"]
    # Either 0 or 1 as strings
    optimizations = ["0", "1"]
    # Use 'k' suffix or integers
    context_lengths = ["4k", "16k"]
    # --- End Configuration ---

    for model_id in model_ids:
        pipeline = RKLLMRemotePipeline(model_id=model_id, platform=platform)
        try:
            pipeline.initialize_and_load()
        except RuntimeError as e:
            print(f"Could not load model {model_id}. Skipping. Error: {e}")
            continue

        hf = HubHelpers(
            platform=platform,
            model_id=model_id,
            lora_id=pipeline.lora_id,
            rkllm_version=pipeline.rkllm_version
        )
        hf.login_to_hf()
        hf.repo_check(model_id)

        print(f"\n--- Starting conversions for {model_id} ---")
        successful_conversions = []
        common_export_path = None
        processed_contexts = set()

        for context in context_lengths:
            try:
                parsed_context, context_name = parse_context_length(context)
                if parsed_context in processed_contexts:
                    print(
                        f"Skipping duplicate context length: {context} (parsed as {parsed_context})")
                    continue
                processed_contexts.add(parsed_context)
            except ValueError as e:
                print(f"Skipping invalid context length: {e}")
                continue

            for qtype in qtypes:
                for hybrid_rate in hybrid_rates:
                    for opt in optimizations:
                        export_path, export_name = pipeline.build_and_export(
                            qtype=qtype,
                            hybrid_rate=hybrid_rate,
                            optimization=int(opt),
                            max_context=parsed_context,
                            context_suffix=context_name
                        )
                        if export_path and export_name:
                            if not common_export_path:
                                common_export_path = export_path
                            successful_conversions.append({
                                'qtype': qtype,
                                'hybrid_rate': hybrid_rate,
                                'opt': opt,
                                'context': context_name,
                                'filename': f"{export_name}.rkllm"
                            })

        if successful_conversions and common_export_path:
            print("\n--- All conversions finished. Preparing for upload. ---")
            repo_name = f"{pipeline.model_name}-{platform}"

            model_import_dir = pipeline.model_dir

            hf.build_card(common_export_path, successful_conversions)

            print("\n--- Unloading model from memory before upload ---")
            del pipeline
            gc.collect()

            hf.upload_to_repo(
                repo_name=repo_name,
                import_path=model_import_dir,
                export_path=common_export_path
            )

        print("\n--- Cleaning up local model files. ---")
        RKLLMRemotePipeline.cleanup_models("./models")

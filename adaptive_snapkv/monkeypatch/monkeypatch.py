from importlib.metadata import version
import warnings
import transformers
from adaptive_snapkv.monkeypatch.fixed_mistral_hijack import fixed_mistral_flash_attn2_forward, prepare_inputs_for_generation_mistral
from adaptive_snapkv.monkeypatch.adaptive_mistral_hijack import adaptive_mistral_flash_attn2_forward,adaptive_MistralModel_forward
def check_version():
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
    version_list = ['4.37']
    warning_flag = True
    for x in version_list:
        if x in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with SnapKV. SnapKV is tested with Transformers version {version_list}.")


def replace_mistral_fixed():
    check_version()
    transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_mistral
    transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = fixed_mistral_flash_attn2_forward

def replace_mistral_adaptive():
    check_version()
    transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_mistral
    transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = adaptive_mistral_flash_attn2_forward
    transformers.models.mistral.modeling_mistral.MistralModel.forward = adaptive_MistralModel_forward

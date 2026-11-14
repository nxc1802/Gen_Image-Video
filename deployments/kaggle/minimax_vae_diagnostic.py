"""Submit/collect a controlled GPU codec diagnosis, never a V3 capability run."""

from deployments.kaggle import native_wave_a

KIND = "minimax_vae_diag"
CASE = "minimax-i2v-astronaut-native-temporal-v2"


def render(*, native_canvas=False, sampler_control=False):
    assert not (native_canvas and sampler_control)
    kind = "minimax_sampler_diag" if sampler_control else "minimax_native_canvas_diag" if native_canvas else KIND
    previous = native_wave_a.VIDEO_BATCHES.get(kind)
    try:
        native_wave_a.VIDEO_BATCHES[kind] = [CASE]
        source = native_wave_a.render_kernel(kind)
    finally:
        if previous is None:
            native_wave_a.VIDEO_BATCHES.pop(kind, None)
        else:
            native_wave_a.VIDEO_BATCHES[kind] = previous
    needle = "'benchmarks.native.wave_'+wave"
    assert source.count(needle) == 1
    source = source.replace(needle, "'benchmarks.native.minimax_vae_diagnostic'")
    source = source.replace("classification='REAL_NATIVE_VALIDATION_ATTEMPT'", "classification='REAL_GPU_SAME_LATENT_VAE_DIAGNOSIS_NOT_V3'")
    if native_canvas or sampler_control:
        assert source.count("*models,*extra]") == 1
        flag = "--sampler-control" if sampler_control else "--native-canvas"
        source = source.replace("*models,*extra]", "*models,*extra," + repr(flag) + "]")
        classification = "REAL_NATIVE_SAMPLER_GPU_DIAGNOSIS_NOT_V3" if sampler_control else "REAL_NATIVE_CANVAS_GPU_DIAGNOSIS_NOT_V3"
        source = source.replace("REAL_GPU_SAME_LATENT_VAE_DIAGNOSIS_NOT_V3", classification)
    compile(source, KIND, "exec")
    assert len(source.encode()) < 1_000_000
    return source


def main():
    import sys

    native_canvas = "--native-canvas" in sys.argv
    sampler_control = "--sampler-control" in sys.argv
    assert not (native_canvas and sampler_control)
    if native_canvas:
        sys.argv.remove("--native-canvas")
    if sampler_control:
        sys.argv.remove("--sampler-control")
    kind = "minimax_sampler_diag" if sampler_control else "minimax_native_canvas_diag" if native_canvas else KIND
    source = render(native_canvas=native_canvas, sampler_control=sampler_control)
    native_wave_a.VIDEO_BATCHES[kind] = [CASE]
    native_wave_a.render_kernel = lambda selected: source if selected == kind else None
    sys.argv = [sys.argv[0], "--kind", kind, *sys.argv[1:]]
    return native_wave_a.main()


if __name__ == "__main__":
    main()

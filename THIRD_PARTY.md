# Third-party components

The model structure and CUDA extension build source are derived from [MambaCD/ChangeMamba](https://github.com/ChenHongruixuan/ChangeMamba), which distributes Apache-2.0-licensed code. The selective scan extension in `third_party/selective_scan/` also descends from [Mamba](https://github.com/state-spaces/mamba), which uses Apache-2.0. The Apache-2.0 license text is supplied in `LICENSE`.

The VSS block implementation in `vendor/` descends from [VMamba](https://github.com/MzeroMiko/VMamba), licensed under MIT. Its permission notice is supplied in `third_party/VMAMBA_LICENSE`.

The `utils/lovasz.py` loss is adapted from [Maxim Berman's LovaszSoftmax implementation](https://github.com/bermanmaxim/LovaszSoftmax), licensed under MIT. Its permission notice is supplied in `third_party/LOVASZ_LICENSE`.

The figures and reported scores are supplied with the anonymous manuscript materials. Dataset images are not redistributed here; obtain them from their respective maintainers.

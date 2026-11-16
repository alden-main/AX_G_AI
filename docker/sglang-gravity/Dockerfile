# learning-unit/L1-16B-A3B(GravityMoE)용 SGLang 이미지입니다.
# 공식 SGLang v0.5.19 이미지에 GravityMoE 등록 패치(gravity.patch)만 적용합니다.
# 모델 카드가 안내하는 trillion-labs/sglang-gravity fork는 v0.5.19 + 이 커밋 하나이므로 결과가 같습니다.
# CUDA 12 이미지를 씁니다(fork 기준, CUDA 13 이미지는 드라이버 580 이상 필요).
ARG SGLANG_BASE_IMAGE=lmsysorg/sglang:v0.5.19-cu129-runtime
FROM ${SGLANG_BASE_IMAGE}

COPY gravity.patch /tmp/gravity.patch
# /sgl-workspace/sglang은 editable 설치 경로입니다. 적용 후 실제로 import되는 sglang에 모델 파일이 있는지 확인합니다.
RUN cd /sgl-workspace/sglang \
    && git apply --verbose /tmp/gravity.patch \
    && rm /tmp/gravity.patch \
    && python3 -c "import pathlib, sglang; p = pathlib.Path(sglang.__file__).parent / 'srt/models/gravity_moe.py'; assert p.is_file(), p; print('GravityMoE:', p)"

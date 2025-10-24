FROM mambaorg/micromamba:1.5.8

WORKDIR /app

COPY environment.yml .

# Create env named vec (or whatever your env name is in environment.yml)
RUN micromamba create -y -n vec -f environment.yml && \
    micromamba clean --all --yes

# Run as non-root user (recommended)
USER $MAMBA_USER

CMD ["micromamba", "run", "-n", "vec", "python", "main.py"]

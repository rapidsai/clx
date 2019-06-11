# Create an image from rapidsai image.
FROM cyberworks/rapidsai:0.7

# Create directory.
RUN mkdir -p /opt/rapidscyber/


# Add scripts and configuration to container.
COPY rapidscyber /opt/rapidscyber/


# Use "bash" as replacement for    "sh"
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Login to virtual environment.
RUN echo "source activate rapids" > ~/.bashrc

# Install required dependencies
RUN source activate rapids && \
    apt-get update && \
    apt-get -y install vim && \
    apt-get -y install supervisor && \
    pip install confluent_kafka mockito pytest
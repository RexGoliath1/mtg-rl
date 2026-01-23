# Forge MTG Headless Simulation Docker Image
# Based on Java 17 for Forge compatibility

FROM maven:3.9-eclipse-temurin-17 AS builder

WORKDIR /build

# Copy the Forge source
COPY forge-repo/ ./

# Remove launch4j and gitchangelog plugins using perl (available in base image)
RUN perl -i -0777 -pe 's/<plugin>\s*<groupId>com\.akathist\.maven\.plugins\.launch4j<\/groupId>.*?<\/plugin>//gs' \
    forge-gui-desktop/pom.xml && \
    perl -i -0777 -pe 's/<plugin>\s*<groupId>se\.bjurr\.gitchangelog<\/groupId>.*?<\/plugin>//gs' \
    forge-gui-desktop/pom.xml

# Build Forge
RUN mvn clean package -DskipTests \
    -pl !forge-gui-android,!forge-gui-ios,!forge-gui-mobile,!forge-gui-mobile-dev,!forge-installer,!adventure-editor,!forge-lda \
    -Dmaven.javadoc.skip=true \
    -Dcheckstyle.skip=true \
    -am

# Find and copy the jar
RUN find /build -name "forge-gui-desktop-*-jar-with-dependencies.jar" -exec cp {} /build/forge.jar \;

# Runtime image - use full JDK for better compatibility
FROM eclipse-temurin:17-jre

# Install Xvfb and dependencies for virtual display
RUN apt-get update && \
    apt-get install -y --no-install-recommends xvfb libxrender1 libxtst6 libxi6 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /forge

# Create required directories
RUN mkdir -p /forge/res \
    /forge/userdata/decks/constructed \
    /forge/userdata/decks/commander \
    /forge/cache

# Copy the built jar
COPY --from=builder /build/forge.jar /forge/forge.jar

# Copy resources (card data, editions, etc.)
COPY --from=builder /build/forge-gui/res/ /forge/res/

# Create forge profile properties pointing to our directories
RUN echo "userDir=/forge/userdata/" > /forge/forge.profile.properties \
    && echo "cacheDir=/forge/cache/" >> /forge/forge.profile.properties \
    && echo "decksDir=/forge/userdata/decks/" >> /forge/forge.profile.properties \
    && echo "decksConstructedDir=/forge/userdata/decks/constructed/" >> /forge/forge.profile.properties

# Copy sample decks
COPY decks/ /forge/userdata/decks/constructed/

# Java options for headless operation
ENV JAVA_OPTS="-Xmx2048m \
    --add-opens java.base/java.lang=ALL-UNNAMED \
    --add-opens java.base/java.util=ALL-UNNAMED \
    --add-opens java.base/java.text=ALL-UNNAMED \
    --add-opens java.base/java.lang.reflect=ALL-UNNAMED \
    --add-opens java.desktop/java.beans=ALL-UNNAMED"

# Create simulation wrapper script that uses xvfb-run
RUN echo '#!/bin/bash\ncd /forge && xvfb-run -a java $JAVA_OPTS -Dsentry.dsn="" -jar forge.jar sim "$@"' > /usr/local/bin/forge-sim \
    && chmod +x /usr/local/bin/forge-sim

# Set working directory for assets lookup
WORKDIR /forge

# Default entrypoint using xvfb-run
ENTRYPOINT ["xvfb-run", "-a", "java", "-Xmx2048m", \
    "--add-opens", "java.base/java.lang=ALL-UNNAMED", \
    "--add-opens", "java.base/java.util=ALL-UNNAMED", \
    "--add-opens", "java.base/java.text=ALL-UNNAMED", \
    "--add-opens", "java.base/java.lang.reflect=ALL-UNNAMED", \
    "--add-opens", "java.desktop/java.beans=ALL-UNNAMED", \
    "-Dsentry.dsn=", \
    "-jar", "/forge/forge.jar", "sim"]

CMD ["-d", "red_aggro.dck", "white_weenie.dck", "-n", "1"]

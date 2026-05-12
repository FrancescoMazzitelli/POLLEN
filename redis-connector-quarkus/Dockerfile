FROM quay.io/quarkus/ubi-quarkus-mandrel-builder-image:jdk-21 AS build
WORKDIR /project
COPY . .
#RUN chmod "a+rwx" /project
RUN ./mvnw package -Dnative -DskipTests

FROM registry.access.redhat.com/ubi9/ubi-minimal:9.6
WORKDIR /work/
RUN chmod "a+rwx" /work

COPY --from=build /project/target/*-runner /work/application

EXPOSE 5600
ENTRYPOINT ["./application", "-Dquarkus.http.host=0.0.0.0"]

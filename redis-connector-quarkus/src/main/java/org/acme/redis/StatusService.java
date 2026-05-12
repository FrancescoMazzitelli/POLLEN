package org.acme.redis;

import io.quarkus.redis.datasource.value.ValueCommands;
import jakarta.enterprise.context.ApplicationScoped;

import io.quarkus.redis.datasource.ReactiveRedisDataSource;
import io.quarkus.redis.datasource.RedisDataSource;
import io.quarkus.redis.datasource.keys.ReactiveKeyCommands;
import org.jboss.logging.Logger;

@ApplicationScoped
public class StatusService {
    private static final Logger LOG = Logger.getLogger(StatusService.class);

    private ReactiveKeyCommands<String> keyCommands;
    private ValueCommands<String, String> countCommands;

    public StatusService(RedisDataSource ds, ReactiveRedisDataSource reactive) {
        countCommands = ds.value(String.class);
        keyCommands = reactive.key();

    }


    String get(String key) {
        String value = countCommands.get(key);
        LOG.info("key = " + key + " value = " + value);
        return value;
    }

    String set(String key, String value) {
        LOG.info("key = " + key + ", value = " + value);
        countCommands.set(key, value.toString());
        String message = "Set " + key + " to " + value;
        return message;
    }

}
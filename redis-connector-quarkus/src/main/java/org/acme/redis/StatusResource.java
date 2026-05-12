package org.acme.redis;

import jakarta.inject.Inject;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.core.MediaType;

import jakarta.ws.rs.core.Response;

@Path("/status")
public class StatusResource {

    @Inject
    StatusService service;

    @GET
    @Path("/{key}")
    public Response get(String key) {
        String status = service.get(key);
        if (status != null && status.equalsIgnoreCase("true")) {
            return Response.ok(status).build();
        }
        return Response.serverError().build();
    }

    @POST
    @Path("/register")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response set(Status status) {
        System.out.println("Register status: " + status.getKey() + " " + status.getValue());
        String message = service.set(status.getKey(), status.getValue());
        if(message == null) {
            return Response.serverError().build();
        }
        return Response.ok(message).build();
    }

}
package water.automl;

import water.api.AbstractRegister;
import water.api.RequestServer;
import water.automl.api.AutoMLBuilderHandler;
import water.automl.api.AutoMLHandler;
import water.automl.api.AutoMLJSONSchemaHandler;

public class Register extends AbstractRegister{
  @Override public void register(String relativeResourcePath) throws ClassNotFoundException {
    // H2O.register("POST /3/AutoMLBuilder", AutoMLBuilderHandler.class, "POST", "automl", "automatically build models");
    // H2O.register("GET /3/AutoML/{automl_id}", AutoMLHandler.class,"refresh", "GET", "refresh the model key");
    // H2O.register("GET /3/AutoMLJSONSchemaHandler", AutoMLJSONSchemaHandler.class, "GET", "getJSONSchema", "Get the json schema for the AutoML input fields.");

    RequestServer.registerEndpoint("automl_build",
            "POST /3/AutoMLBuilder", AutoMLBuilderHandler.class, "build",
            "Unlock all keys in the H2O distributed K/V store, to attempt to recover from a crash.");
    RequestServer.registerEndpoint("automl_refresh",
            "GET /3/AutoML/{automl_id}", AutoMLHandler.class, "refresh",
            "Refresh the model key.");
    RequestServer.registerEndpoint("automl_schema",
            "GET /3/AutoMLJSONSchemaHandler", AutoMLJSONSchemaHandler.class, "getJSONSchema",
            "Get the json schema for the AutoML input fields.");

  }
}

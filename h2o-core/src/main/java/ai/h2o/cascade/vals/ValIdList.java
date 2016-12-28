package ai.h2o.cascade.vals;

import ai.h2o.cascade.asts.AstIdList;
import ai.h2o.cascade.core.IdList;


/**
 * List of (unevaluated) ids.
 */
public class ValIdList extends Val {
  private IdList value;


  public ValIdList(IdList d) {
    value = d;
  }

  @Override public Type type() {
    return Type.IDS;
  }

  @Override public String toString() {
    return new AstIdList(value).str();
  }


  //--------------------------------------------------------------------------------------------------------------------
  // Value representations
  //--------------------------------------------------------------------------------------------------------------------

  @Override public boolean isIds() {
    return true;
  }

  @Override public IdList getIds() {
    return value;
  }
}
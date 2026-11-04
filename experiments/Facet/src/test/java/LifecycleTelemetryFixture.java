public final class LifecycleTelemetryFixture {
  public static void main(String[] args) {
    AdditionalRegressionTest.terminal(
        FacetSemanticTest.body(400, 300, 0, 0), FacetSemanticTest.RULES);
    System.out.println("PASS two terminal orders, duplicate completion, cold reused process, four "
                       + "unique game-round rows");
  }
}

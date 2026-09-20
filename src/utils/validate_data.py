import pandas as pd
import great_expectations as gx
from typing import List, Tuple


def _build_suite() -> "gx.ExpectationSuite":
    """
    Build the Great Expectations suite for the raw Telco Customer Churn dataset.

    The suite encodes the data contract the classifier depends on: required
    columns, allowed categorical values, numeric ranges (business bounds) and
    one cross-field consistency rule. Every check here has a business reason.
    """
    suite = gx.ExpectationSuite(name="telco_churn_suite")
    e = gx.expectations

    # === SCHEMA VALIDATION - ESSENTIAL COLUMNS ===
    # Customer id (needed downstream by the business), demographics, service
    # features and the financial features that drive churn.
    for column in [
        "customerID",
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "InternetService",
        "Contract",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]:
        suite.add_expectation(e.ExpectColumnToExist(column=column))

    # === NULL CHECKS ON CRITICAL COLUMNS ===
    for column in ["customerID", "tenure", "MonthlyCharges"]:
        suite.add_expectation(e.ExpectColumnValuesToNotBeNull(column=column))

    # === BUSINESS LOGIC VALIDATION ===
    # Categorical values must be within the sets the encoders know about,
    # otherwise train/serve encoding would silently drift.
    suite.add_expectation(
        e.ExpectColumnValuesToBeInSet(column="gender", value_set=["Male", "Female"])
    )
    suite.add_expectation(
        e.ExpectColumnValuesToBeInSet(column="Partner", value_set=["Yes", "No"])
    )
    suite.add_expectation(
        e.ExpectColumnValuesToBeInSet(column="Dependents", value_set=["Yes", "No"])
    )
    suite.add_expectation(
        e.ExpectColumnValuesToBeInSet(column="PhoneService", value_set=["Yes", "No"])
    )
    suite.add_expectation(
        e.ExpectColumnValuesToBeInSet(
            column="Contract", value_set=["Month-to-month", "One year", "Two year"]
        )
    )
    suite.add_expectation(
        e.ExpectColumnValuesToBeInSet(
            column="InternetService", value_set=["DSL", "Fiber optic", "No"]
        )
    )

    # === NUMERIC RANGE / STATISTICAL VALIDATION ===
    # tenure: 0-120 months (<= 10 years on this product), charges within the
    # observed business ranges. Catches corrupted/negative data entry.
    suite.add_expectation(
        e.ExpectColumnValuesToBeBetween(column="tenure", min_value=0, max_value=120)
    )
    suite.add_expectation(
        e.ExpectColumnValuesToBeBetween(column="MonthlyCharges", min_value=0, max_value=200)
    )
    suite.add_expectation(e.ExpectColumnValuesToBeBetween(column="TotalCharges", min_value=0))

    # === DATA CONSISTENCY CHECK ===
    # Total charges should generally be >= monthly charges; the 5% allowance
    # covers brand-new customers / edge cases while still catching entry errors.
    suite.add_expectation(
        e.ExpectColumnPairValuesAToBeGreaterThanB(
            column_A="TotalCharges",
            column_B="MonthlyCharges",
            or_equal=True,
            mostly=0.95,
        )
    )

    return suite


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Telco Customer Churn dataset using Great Expectations.

    This function implements critical data quality checks that must pass before model training.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.

    Implemented with the Great Expectations 1.x fluent API using an *ephemeral*
    (in-memory) context, so no `great_expectations/` project directory is written
    to disk - important for reproducible pipelines and containers.

    Args:
        df: Raw pandas DataFrame (as loaded from the source CSV).

    Returns:
        Tuple[bool, List[str]]: (True if every expectation passed, failed expectation types)

    """
    print("🔍 Starting data validation with Great Expectations...")

    # TotalCharges arrives as a string column with blank values (11 rows in the
    # Telco dataset) -> coerce it on a copy so numeric expectations can actually
    # be evaluated. The permanent fix happens later in preprocess_data().
    df = df.copy()
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # === BUILD BATCH (GE 1.x FLUENT API) ===
    context = gx.get_context(mode="ephemeral")
    datasource = context.data_sources.add_pandas("telco_pandas")
    asset = datasource.add_dataframe_asset("telco_churn_raw")
    batch_definition = asset.add_batch_definition_whole_dataframe("whole_dataframe")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})
    suite = context.suites.add(_build_suite())

    # === RUN VALIDATION SUITE ===
    print("   ⚙️  Running complete validation suite...")
    results = batch.validate(suite)

    # === PROCESS RESULTS ===
    # Extract failed expectations for detailed error reporting
    failed_expectations = []
    for r in results.results:
        if not r.success:
            config = r.expectation_config
            failed_expectations.append(
                getattr(config, "type", None) or config.get("expectation_type")
            )

    # Print validation summary
    total_checks = len(results.results)
    passed_checks = sum(1 for r in results.results if r.success)
    failed_checks = total_checks - passed_checks

    if results.success:
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"❌ Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"   Failed expectations: {failed_expectations}")

    return bool(results.success), failed_expectations

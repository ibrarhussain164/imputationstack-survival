# Imputation-Based Survival Stacking Pipeline
# Imputation_Stacking_Survival_Pipeline.R
#
# Within each fixed survival classifier (CoxPH / GBM / SurvKNN), compares
# classical missing-data handlers, single MICE engines, and ImpStack_MICE.
#
# ImpStack_MICE (per-copy stacking, then average):
#   for each imputation copy k = 1..m:
#     Level-0 = NORM/PMM/CART/MIDASTOUCH/RF fits on copy k (OOF + test LPs)
#     meta    = unpenalised CoxPH on those Level-0 LPs
#     store stacked LP / S(t) for copy k
#   final ImpStack = average of the m stacked predictions.
#
# Design notes
#   - Within-classifier Level-0 only (CoxPH/GBM/SurvKNN are never mixed at Level-0).
#   - Stack library: NORM, PMM, CART, MIDASTOUCH, RF; meta-learner: CoxPH.
#   - Artificial missingness on TRAIN covariates only; TEST stays complete.
#   - MICE uses the Nelson-Aalen cumulative hazard and the event indicator as
#     predictors (White & Royston); observed time is auxiliary; time/event are
#     never imputed.
#   - Level-0 hyperparameters are fixed (no inner CV tuning).
#   - Deterministic seeds keyed on dataset x split x rate x mechanism.
#   - MIE-SE reports Harrell C / CalSlope only; IBS is omitted (heterogeneous
#     Level-0 scales).
#
# Methods evaluated per classifier block
#   MeanSI / RFI / MissInd / CCA        classical handlers
#   NORM / PMM / CART / MIDASTOUCH / RF one MICE engine, m=5, averaged
#   ImpStack_MICE                       stack each of m copies, then average
#   MICE_SE (Aleryani)                  cross-classifier stack; L0 = K*m scores
#
# Metrics: Harrell C, IBS, calibration slope. See EVALUATION.md / METRICS.md.

# ---- 0. Packages (do not auto-install by default) ----
INSTALL_MISSING_PKGS <- FALSE  # set TRUE only for local bootstrap installs
needed <- c("mice", "missForest", "survival", "gbm",
            "bnnSurvival", "caret",
            "dplyr", "tidyr", "ggplot2", "purrr", "readr", "scales",
            "randomForest")
missing_pkgs <- needed[!needed %in% rownames(installed.packages())]
if (length(missing_pkgs) > 0) {
  if (isTRUE(INSTALL_MISSING_PKGS)) {
    message("Installing missing packages: ", paste(missing_pkgs, collapse = ", "))
    install.packages(missing_pkgs, repos = "https://cloud.r-project.org")
  } else {
    stop("Missing required packages (install manually or set INSTALL_MISSING_PKGS=TRUE): ",
         paste(missing_pkgs, collapse = ", "))
  }
}
suppressPackageStartupMessages({
  library(mice); library(missForest)
  library(survival); library(gbm)
  library(bnnSurvival)
  library(caret)
  library(dplyr); library(tidyr); library(ggplot2)
  library(purrr); library(readr); library(scales); library(randomForest)
})

GLOBAL_SEED <- 2024L
set.seed(GLOBAL_SEED)

# ---- 1. Settings (freeze for a reported run; bump RUN_ID if any change) ----
N_SPLITS   <- 10L
MISS_RATES <- seq(0.10, 0.50, by = 0.10)
MAXIT_MICE <- 5L
M_IMP      <- 5L
CV_FOLDS   <- 5L
TRAIN_FRAC <- 0.70
MISS_MECHS <- c("MAR", "MCAR")

SURV_NAMES  <- c("CoxPH", "GBM", "SurvKNN")
IMP_ENGINES <- c("norm", "pmm", "cart", "midastouch", "rf")

SI_BASELINES <- c("MeanSI", "RFI", "MissInd", "CCA")

# Multi-dataset evaluation design
REF_BASELINE    <- "MeanSI"   # paired deltas / win rates vs mean fill
PRIMARY_METRICS <- c("Cindex", "IBS", "CalSlope")

# Stack library: 5 MICE engines named directly (PMM/RF/CART/NORM/MIDASTOUCH)
# mice() engine key (lowercase) → Method label used in results/plots
ENGINE_METHOD <- c(
  pmm = "PMM",
  rf = "RF",
  cart = "CART",
  norm = "NORM",
  midastouch = "MIDASTOUCH"
)
STACK_MICE_LIB <- unname(ENGINE_METHOD[IMP_ENGINES])
META_LEARNER   <- "CoxPH"

# Fixed Level-0 defaults (no inner CV).
# Isolates missing-data handlers under a common learner and equal compute budget.
# OOF CV (CV_FOLDS) builds ImpStack meta-features only; it does not tune HPs.
# See DESIGN_NOTES.md.
GBM_N_TREES    <- 200L
KNN_K          <- 20L   # Lowsky / bnnSurvival neighbors
KNN_N_BAG      <- 1L    # non-bagged Lowsky form (bagging off)

# Plot / table order (Stacking first). Keys = literal Method codes.
METHOD_ORDER <- c(
  "ImpStack_MICE",
  "PMM", "RF", "CART", "NORM", "MIDASTOUCH",
  "MissInd", "RFI", "MeanSI", "CCA",
  "MICE_SE"
)

# Display names for legends / axes (must align with METHOD_ORDER keys)
METHOD_DISPLAY <- c(
  ImpStack_MICE     = "Stacking",
  PMM               = "PMM",
  RF                = "RF",
  CART              = "CART",
  NORM              = "NORM",
  MIDASTOUCH        = "MIDASTOUCH",
  MissInd           = "Missing Indicator",
  RFI               = "MissForest",
  MeanSI            = "Mean Imputation",
  CCA               = "CCA",
  MICE_SE           = "MIE-SE"
)

# Colours keyed by literal Method codes (same keys as METHOD_ORDER)
METHOD_COLORS <- c(
  ImpStack_MICE     = "blue",
  PMM               = "grey",
  RF                = "lightgreen",
  CART              = "lightpink",
  NORM              = "lightblue",
  MIDASTOUCH        = "brown",
  MissInd           = "darkorange",
  RFI               = "darkgreen",
  MeanSI            = "red",
  CCA               = "purple",
  MICE_SE           = "darkblue"
)

METHOD_CATALOG <- data.frame(
  Method = METHOD_ORDER,
  Display = unname(METHOD_DISPLAY[METHOD_ORDER]),
  Role = c(
    "stacking_mice",
    rep("level0_engine", 5L),
    rep("baseline", 4L),
    "stacking_cross_clf"
  ),
  Definition = c(
    "ImpStack_MICE (Stacking): per-copy CoxPH meta on PMM/RF/CART/NORM/MIDASTOUCH (k=1..m), then average m stacks.",
    "MICE method=pmm, m=5 on train; predict on complete test; average LP/S(t).",
    "MICE method=rf, m=5 on train; predict on complete test; average LP/S(t).",
    "MICE method=cart, m=5 on train; predict on complete test; average LP/S(t).",
    "MICE method=norm, m=5 on train; predict on complete test; average LP/S(t).",
    "MICE method=midastouch, m=5 on train; predict on complete test; average LP/S(t).",
    "Train: mean fill + missingness indicators. Test: indicators all zero.",
    "missForest on train only; test is complete (unchanged).",
    "Train: mean-fill with training-column means. Test: complete (unchanged).",
    "Fit on complete training cases; evaluate on all (complete) test cases.",
    "MICE_SE / MIE-SE (Aleryani original): MI (pmm) + heterogeneous Level-0; concatenate all K*m OOF/test score columns; meta = listed Classifier; Harrell C/CalSlope only (IBS skipped: mixed score scales)."
  ),
  stringsAsFactors = FALSE
)

RUN_ID         <- "impstack_v42"
# Resolve project root from script path when possible; else current working directory.
PROJECT_ROOT <- local({
  ca <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", ca[grepl("^--file=", ca)])
  if (length(f) == 1L && nzchar(f)) {
    return(dirname(normalizePath(f, winslash = "/", mustWork = FALSE)))
  }
  normalizePath(getwd(), winslash = "/", mustWork = FALSE)
})
OUTPUT_DIR     <- file.path(PROJECT_ROOT, paste0("surv_", RUN_ID))
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
OUTPUT_DIR     <- normalizePath(OUTPUT_DIR, winslash = "/", mustWork = TRUE)
CHECKPOINT_DIR <- file.path(OUTPUT_DIR, "checkpoints")
FIG_DIR        <- file.path(OUTPUT_DIR, "figures")
DATA_DIR       <- file.path(path.expand("~"), "survival_data")
DATA_DIR2      <- file.path(PROJECT_ROOT, "survival_data", "data")
REPRO_DIR      <- file.path(OUTPUT_DIR, "reproducibility")
dir.create(CHECKPOINT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIG_DIR,        recursive = TRUE, showWarnings = FALSE)
dir.create(REPRO_DIR,      recursive = TRUE, showWarnings = FALSE)
write_csv(METHOD_CATALOG, file.path(OUTPUT_DIR, "method_catalog.csv"))
cat("Output directory:", OUTPUT_DIR, "\n")
cat("Figures directory:", FIG_DIR, "\n")
cat("Reproducibility :", REPRO_DIR, "\n")

# ---- Configuration checks ----
assert_config <- function() {
  stopifnot(all(IMP_ENGINES %in% names(ENGINE_METHOD)))
  stopifnot(all(STACK_MICE_LIB %in% METHOD_ORDER))
  stopifnot(all(METHOD_ORDER %in% names(METHOD_DISPLAY)))
  stopifnot(all(METHOD_ORDER %in% names(METHOD_COLORS)))
  stopifnot(nrow(METHOD_CATALOG) == length(METHOD_ORDER))
  stopifnot(identical(as.character(METHOD_CATALOG$Method), METHOD_ORDER))
  stopifnot(REF_BASELINE %in% METHOD_ORDER)
  stopifnot(all(PRIMARY_METRICS %in% c("Cindex", "IBS", "CalSlope")))
  stopifnot(TRAIN_FRAC > 0.5, TRAIN_FRAC < 1)
  stopifnot(as.integer(M_IMP) >= 2L, as.integer(CV_FOLDS) >= 2L, as.integer(N_SPLITS) >= 1L)
  invisible(TRUE)
}
assert_config()

capture_package_versions <- function() {
  pkgs <- sort(unique(c(needed, "base")))
  vers <- vapply(pkgs, function(p) {
    if (identical(p, "base")) as.character(getRversion())
    else as.character(utils::packageVersion(p))
  }, character(1))
  data.frame(package = pkgs, version = vers, stringsAsFactors = FALSE)
}

git_commit_hash <- function() {
  tryCatch({
    out <- system2("git", c("-C", PROJECT_ROOT, "rev-parse", "HEAD"),
                   stdout = TRUE, stderr = FALSE)
    if (length(out) == 1L && nzchar(out)) out else NA_character_
  }, error = function(e) NA_character_)
}

write_analysis_config <- function(path = file.path(REPRO_DIR, "analysis_config.txt")) {
  lines <- c(
    paste0("RUN_ID=", RUN_ID),
    paste0("GLOBAL_SEED=", GLOBAL_SEED),
    paste0("N_SPLITS=", N_SPLITS),
    paste0("TRAIN_FRAC=", TRAIN_FRAC),
    paste0("MISS_RATES=", paste(MISS_RATES, collapse = ",")),
    paste0("MISS_MECHS=", paste(MISS_MECHS, collapse = ",")),
    paste0("M_IMP=", M_IMP),
    paste0("MAXIT_MICE=", MAXIT_MICE),
    paste0("CV_FOLDS=", CV_FOLDS),
    paste0("SURV_NAMES=", paste(SURV_NAMES, collapse = ",")),
    paste0("IMP_ENGINES=", paste(IMP_ENGINES, collapse = ",")),
    paste0("STACK_MICE_LIB=", paste(STACK_MICE_LIB, collapse = ",")),
    paste0("META_LEARNER=", META_LEARNER),
    paste0("METHOD_ORDER=", paste(METHOD_ORDER, collapse = ",")),
    paste0("REF_BASELINE=", REF_BASELINE),
    paste0("PRIMARY_METRICS=", paste(PRIMARY_METRICS, collapse = ",")),
    paste0("DATASETS=", paste(vapply(ALL_DATASETS, `[[`, "", "name"), collapse = ",")),
    paste0("FLCHAIN_N_SUB=", FLCHAIN_N_SUB),
    paste0("FLCHAIN_SUB_SEED=", FLCHAIN_SUB_SEED),
    paste0("GBM_N_TREES=", GBM_N_TREES),
    paste0("KNN_K=", KNN_K),
    paste0("KNN_N_BAG=", KNN_N_BAG),
    paste0("DATA_DIR=", DATA_DIR),
    paste0("PROJECT_ROOT=", PROJECT_ROOT),
    paste0("OUTPUT_DIR=", OUTPUT_DIR),
    paste0("git_commit=", git_commit_hash()),
    paste0("timestamp_start=", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
    paste0("R_version=", R.version.string),
    paste0("platform=", R.version$platform)
  )
  writeLines(lines, path)
  writeLines(lines, file.path(OUTPUT_DIR, "analysis_config.txt"))
  invisible(path)
}

write_reproducibility_start <- function() {
  write_analysis_config()
  write_csv(capture_package_versions(),
            file.path(REPRO_DIR, "package_versions.csv"))
  sink(file.path(REPRO_DIR, "sessionInfo_start.txt"))
  print(sessionInfo())
  sink()
  # Copy this script into the run folder for archival
  script_src <- file.path(PROJECT_ROOT, "Imputation_Stacking_Survival_Pipeline.R")
  if (file.exists(script_src)) {
    file.copy(script_src,
              file.path(REPRO_DIR, "Imputation_Stacking_Survival_Pipeline.R"),
              overwrite = TRUE)
  }
  for (aux in c("DESIGN_NOTES.md", "README.md", "method_plot_style.R")) {
    p <- file.path(PROJECT_ROOT, aux)
    if (file.exists(p)) file.copy(p, file.path(REPRO_DIR, aux), overwrite = TRUE)
  }
  invisible(TRUE)
}

write_reproducibility_end <- function(results_df = NULL) {
  sink(file.path(REPRO_DIR, "sessionInfo_end.txt"))
  print(sessionInfo())
  sink()
  end_note <- c(
    paste0("timestamp_end=", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
    paste0("n_result_rows=", if (!is.null(results_df)) nrow(results_df) else NA),
    paste0("git_commit=", git_commit_hash())
  )
  writeLines(end_note, file.path(REPRO_DIR, "run_end.txt"))
  # File checksums for key outputs
  key_files <- c(
    "impstack_raw_results.csv",
    "impstack_grand_means.csv",
    "impstack_dataset_means.csv",
    "method_catalog.csv",
    "analysis_config.txt"
  )
  checksums <- lapply(key_files, function(fn) {
    fp <- file.path(OUTPUT_DIR, fn)
    if (!file.exists(fp)) return(NULL)
    data.frame(
      file = fn,
      bytes = file.info(fp)$size,
      md5 = as.character(tools::md5sum(fp)),
      stringsAsFactors = FALSE
    )
  })
  checksums <- dplyr::bind_rows(checksums)
  if (nrow(checksums) > 0)
    write_csv(checksums, file.path(REPRO_DIR, "output_checksums.csv"))
  invisible(TRUE)
}

# ---- Computation structure + timing logs (saved for reproducibility / cost) ----
TIMING_DIR <- file.path(OUTPUT_DIR, "timing")
dir.create(TIMING_DIR, recursive = TRUE, showWarnings = FALSE)

build_computation_structure <- function() {
  n_clf <- length(SURV_NAMES)
  n_eng <- length(IMP_ENGINES)
  n_base <- length(SI_BASELINES)
  n_meth_within <- length(METHOD_ORDER) - 1L  # exclude MICE_SE (cross-clf)
  # Approximate Level-0 survival fits PER ARTIFICIAL SPLIT (order-of-magnitude)
  # Within each classifier block:
  #   SI baselines: ~4 fits (MeanSI,RFI,MissInd,CCA)
  #   PMM/RF/... : n_eng * m full fits (avg over m) + n_eng * m * CV_FOLDS OOF if stacking path
  #   ImpStack_MICE: for each of m copies: n_eng*(CV_FOLDS OOF + 1 test) + 1 meta
  fits_si_per_clf <- n_base
  fits_indiv_per_clf <- n_eng * as.integer(M_IMP)          # Indiv avg path
  fits_impstack_per_clf <- as.integer(M_IMP) * (
    n_eng * (as.integer(CV_FOLDS) + 1L) + 1L)             # OOF+test+meta per copy
  fits_within_total <- n_clf * (fits_si_per_clf + fits_indiv_per_clf +
                                  fits_impstack_per_clf)
  # MICE_SE (original Aleryani): K*m*(CV_FOLDS OOF + 1 test) + K metas
  fits_mice_se <- n_clf * as.integer(M_IMP) * (as.integer(CV_FOLDS) + 1L) +
    n_clf
  fits_per_split_approx <- fits_within_total + fits_mice_se
  
  n_ds <- length(ALL_DATASETS)
  n_mech <- length(MISS_MECHS)
  n_rate <- length(MISS_RATES)
  n_splits_art <- n_ds * n_mech * n_rate * N_SPLITS  # if all artificial
  list(
    run_id = RUN_ID,
    created_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
    r_version = R.version.string,
    platform = R.version$platform,
    design = list(
      classifiers = SURV_NAMES,
      n_classifiers = n_clf,
      baselines = SI_BASELINES,
      mice_engines = IMP_ENGINES,
      methods = METHOD_ORDER,
      meta_learner_impstack = META_LEARNER,
      mice_se_style = "Aleryani_original_Km_columns",
      mice_se_l0_columns = n_clf * as.integer(M_IMP),
      metrics = c("Cindex", "IBS", "CalSlope"),
      mice_se_metrics = c("Cindex", "CalSlope"),
      missingness = "train_only_test_complete",
      mechanisms = MISS_MECHS,
      miss_rates = MISS_RATES,
      n_splits = N_SPLITS,
      train_frac = TRAIN_FRAC,
      m_imp = as.integer(M_IMP),
      mice_maxit = MAXIT_MICE,
      cv_folds_oof = CV_FOLDS,
      knn_k = KNN_K,
      knn_n_bag = KNN_N_BAG,
      gbm_n_trees = GBM_N_TREES
    ),
    datasets = vapply(ALL_DATASETS, function(d) d$name, character(1)),
    approx_fit_counts_per_artificial_split = list(
      note = "Order-of-magnitude survival model fits (not wall-clock). MICE itself extra.",
      per_classifier_SI = fits_si_per_clf,
      per_classifier_mice_engines = fits_indiv_per_clf,
      per_classifier_ImpStack = fits_impstack_per_clf,
      all_within_classifier_blocks = fits_within_total,
      MICE_SE_Km_plus_meta = fits_mice_se,
      total_approx = fits_per_split_approx
    ),
    approx_grid_size = list(
      n_datasets = n_ds,
      n_mechanisms = n_mech,
      n_rates = n_rate,
      n_splits = N_SPLITS,
      artificial_cells_if_all_ds = n_splits_art,
      result_rows_per_split = n_clf * length(METHOD_ORDER)
    )
  )
}

write_computation_structure <- function(struct, dir = TIMING_DIR) {
  # JSON-ish via dput + readable text (no extra JSON package required)
  saveRDS(struct, file.path(dir, "computation_structure.rds"))
  utils::capture.output(str(struct, max.level = 3, give.attr = FALSE),
                        file = file.path(dir, "computation_structure_str.txt"))
  lines <- c(
    paste0("RUN_ID: ", struct$run_id),
    paste0("Created: ", struct$created_at),
    paste0("R: ", struct$r_version),
    paste0("Platform: ", struct$platform),
    "",
    "=== DESIGN ===",
    paste0("Classifiers (", struct$design$n_classifiers, "): ",
           paste(struct$design$classifiers, collapse = " | ")),
    paste0("Methods: ", paste(struct$design$methods, collapse = " | ")),
    paste0("MICE engines (m=", struct$design$m_imp, ", maxit=",
           struct$design$mice_maxit, "): ",
           paste(struct$design$mice_engines, collapse = ", ")),
    paste0("ImpStack meta: ", struct$design$meta_learner_impstack),
    paste0("MICE_SE: ", struct$design$mice_se_style,
           " | L0 cols = ", struct$design$mice_se_l0_columns),
    paste0("OOF CV folds: ", struct$design$cv_folds_oof),
    paste0("SurvKNN: k=", struct$design$knn_k,
           " n_bag=", struct$design$knn_n_bag),
    paste0("Missingness: ", struct$design$missingness),
    paste0("Mechanisms: ", paste(struct$design$mechanisms, collapse = ", ")),
    paste0("Rates: ", paste(sprintf("%.0f%%", 100 * struct$design$miss_rates),
                            collapse = ", ")),
    paste0("Splits/cell: ", struct$design$n_splits),
    paste0("Train frac: ", struct$design$train_frac),
    "",
    "=== DATASETS ===",
    paste(struct$datasets, collapse = ", "),
    "",
    "=== APPROX SURVIVAL FITS / ARTIFICIAL SPLIT ===",
    paste0("  SI baselines / clf: ",
           struct$approx_fit_counts_per_artificial_split$per_classifier_SI),
    paste0("  Indiv avg / clf: ",
           struct$approx_fit_counts_per_artificial_split$per_classifier_mice_engines),
    paste0("  ImpStack / clf: ",
           struct$approx_fit_counts_per_artificial_split$per_classifier_ImpStack),
    paste0("  Within-clf total: ",
           struct$approx_fit_counts_per_artificial_split$all_within_classifier_blocks),
    paste0("  MICE_SE: ",
           struct$approx_fit_counts_per_artificial_split$MICE_SE_Km_plus_meta),
    paste0("  TOTAL ~ ",
           struct$approx_fit_counts_per_artificial_split$total_approx,
           " survival fits / split (+ MICE imputation cost)"),
    "",
    "=== GRID SIZE ===",
    paste0("Datasets: ", struct$approx_grid_size$n_datasets),
    paste0("Artificial cells (ds×mech×rate×split) if all artificial: ",
           struct$approx_grid_size$artificial_cells_if_all_ds),
    paste0("Result rows / split: ", struct$approx_grid_size$result_rows_per_split),
    "",
    "Timing CSVs written during/after the run in this folder:",
    "  timing_per_split.csv",
    "  timing_by_dataset.csv",
    "  timing_summary.csv"
  )
  writeLines(lines, file.path(dir, "computation_structure.txt"))
  # also copy to OUTPUT_DIR root for easy find
  writeLines(lines, file.path(OUTPUT_DIR, "computation_structure.txt"))
  file.copy(file.path(dir, "computation_structure.rds"),
            file.path(OUTPUT_DIR, "computation_structure.rds"),
            overwrite = TRUE)
  invisible(TRUE)
}

# COMP_STRUCT written after ALL_DATASETS is defined (section 3 / main loop).
timing_split_log <- list()
timing_dataset_log <- list()

# ---- 2. Seed helper ----
make_split_seed <- function(sp, ds_name, miss_rate = 0, mech = "NONE") {
  mech_offset <- switch(mech, MAR = 9999L, MCAR = 19999L, NATURAL = 29999L, 0L)
  rate_part <- if (is.finite(miss_rate)) as.integer(round(miss_rate * 1000)) else 777L
  sp * 137L + rate_part + nchar(ds_name) + mech_offset
}

# ---- 3. Dataset loaders ----
make_surv_ds <- function(X_num, time, event, keep_na_predictors = FALSE) {
  df <- data.frame(X_num, check.names = FALSE)
  df$time  <- as.numeric(time)
  df$event <- as.integer(event)
  df <- df[complete.cases(df[, c("time", "event")]) & df$time > 0, ]
  pred_cols <- setdiff(names(df), c("time", "event"))
  num_ok <- pred_cols[vapply(df[, pred_cols, drop = FALSE], is.numeric, logical(1))]
  df <- df[, c(num_ok, "time", "event"), drop = FALSE]
  if (!keep_na_predictors) {
    df <- df[stats::complete.cases(df), , drop = FALSE]
  }
  df
}

load_csv <- function(filename, keep_na_predictors = FALSE) {
  candidates <- c(file.path(DATA_DIR, filename), file.path(DATA_DIR2, filename))
  fpath <- Find(file.exists, candidates)
  if (is.null(fpath)) stop(sprintf("CSV not found: %s", filename))
  d <- read.csv(fpath, stringsAsFactors = FALSE, na.strings = c("", "NA", "NaN", "?"))
  pred_cols <- setdiff(names(d), c("time", "event"))
  X <- d[, pred_cols, drop = FALSE]
  X <- X[, vapply(X, is.numeric, logical(1)), drop = FALSE]
  make_surv_ds(X, d$time, d$event, keep_na_predictors = keep_na_predictors)
}

# FLchain: fixed event-stratified subsample (computational feasibility; Methods).
# Full public flchain is ~7870; we use n = FLCHAIN_N_SUB with seed FLCHAIN_SUB_SEED.
FLCHAIN_N_SUB    <- 2000L
FLCHAIN_SUB_SEED <- 2024L
load_flchain_subsample <- function(n_sub = FLCHAIN_N_SUB,
                                   seed = FLCHAIN_SUB_SEED,
                                   keep_na_predictors = FALSE) {
  df <- load_csv("FLchain.csv", keep_na_predictors = keep_na_predictors)
  n <- nrow(df)
  if (n <= n_sub) {
    message(sprintf("FLchain: using full n=%d (<= subsample size %d)", n, n_sub))
    return(df)
  }
  set.seed(seed)
  idx_e1 <- which(df$event == 1L)
  idx_e0 <- which(df$event == 0L)
  # Preserve observed event rate (rounded); ensure >=1 of each if possible
  n_e1 <- max(1L, min(length(idx_e1), as.integer(round(n_sub * length(idx_e1) / n))))
  n_e0 <- n_sub - n_e1
  if (n_e0 > length(idx_e0)) {
    n_e0 <- length(idx_e0)
    n_e1 <- n_sub - n_e0
  }
  if (n_e1 > length(idx_e1)) {
    n_e1 <- length(idx_e1)
    n_e0 <- n_sub - n_e1
  }
  take <- c(sample(idx_e1, n_e1), sample(idx_e0, n_e0))
  out <- df[sort(take), , drop = FALSE]
  rownames(out) <- NULL
  attr(out, "flchain_subsample") <- list(
    n_full = n, n_sub = nrow(out), seed = seed,
    n_events = sum(out$event == 1L),
    event_rate = mean(out$event == 1L)
  )
  message(sprintf(
    "FLchain subsample: n=%d / %d (seed=%d; events=%d, rate=%.3f)",
    nrow(out), n, seed, sum(out$event == 1L), mean(out$event == 1L)
  ))
  out
}

# PBC with *natural* missingness retained (no complete-case drop on predictors)
# (optional helper; not in ALL_DATASETS)
load_pbc_natural <- function() {
  load_csv("PBC.csv", keep_na_predictors = TRUE)
}

obs_missing_rate <- function(df) {
  pred <- setdiff(names(df), c("time", "event"))
  if (length(pred) == 0) return(NA_real_)
  mean(is.na(as.matrix(df[, pred, drop = FALSE])))
}

# Artificial missingness benchmarks (9 datasets; FLchain = n=2000 subsample)
ARTIFICIAL_DATASETS <- list(
  list(name = "WHAS500",      mode = "artificial",
       loader = function() load_csv("WHAS500.csv")),
  list(name = "GBSG2",        mode = "artificial",
       loader = function() load_csv("GBSG2.csv")),
  list(name = "MGUS2",        mode = "artificial",
       loader = function() load_csv("MGUS2.csv")),
  list(name = "WIHS",         mode = "artificial",
       loader = function() load_csv("WIHS.csv")),
  list(name = "PBC",          mode = "artificial",
       loader = function() load_csv("PBC.csv")),
  list(name = "Colon",        mode = "artificial",
       loader = function() load_csv("Colon.csv")),
  list(name = "HeartFailure", mode = "artificial",
       loader = function() load_csv("HeartFailure.csv")),
  list(name = "ACTG175",      mode = "artificial",
       loader = function() load_csv("ACTG175.csv")),
  list(name = "FLchain",      mode = "artificial",
       loader = function() load_flchain_subsample())
)

# SUPPORT2 / other natural datasets removed from this pipeline.
ALL_DATASETS <- ARTIFICIAL_DATASETS

# Capture frozen settings + session AFTER dataset list is known
write_reproducibility_start()

# ---- 4. Missing-data injection ----
induce_missing <- function(X, rate, mech = "MAR",
                           time_vec = NULL, event_vec = NULL, seed = NULL) {
  if (rate == 0) return(X)
  if (!is.null(seed)) set.seed(seed)
  n <- nrow(X); p <- ncol(X); X_miss <- as.data.frame(X)
  if (toupper(mech) == "MAR") {
    beta_mar <- log(9)
    for (j in seq_len(p)) {
      others   <- setdiff(seq_len(p), j)
      aux_cols <- list()
      if (length(others) > 0) {
        Xoth <- scale(as.matrix(X[, others, drop = FALSE]))
        Xoth[!is.finite(Xoth)] <- 0
        aux_cols[[length(aux_cols) + 1]] <- Xoth
      }
      if (!is.null(time_vec) && length(time_vec) == n) {
        lt <- scale(log(pmax(as.numeric(time_vec), 1e-6)))
        lt[!is.finite(lt)] <- 0
        aux_cols[[length(aux_cols) + 1]] <- matrix(lt, ncol = 1)
      }
      if (!is.null(event_vec) && length(event_vec) == n) {
        ev <- scale(as.numeric(event_vec))
        ev[!is.finite(ev)] <- 0
        aux_cols[[length(aux_cols) + 1]] <- matrix(ev, ncol = 1)
      }
      if (length(aux_cols) == 0) {
        X_miss[sample(n, max(1L, round(rate * n))), j] <- NA
        next
      }
      mar_score <- rowMeans(do.call(cbind, aux_cols))
      alpha_j <- tryCatch(
        uniroot(function(a) mean(plogis(a + beta_mar * mar_score)) - rate,
                interval = c(-50, 50), tol = 1e-8)$root,
        error = function(e) qlogis(rate))
      X_miss[runif(n) < plogis(alpha_j + beta_mar * mar_score), j] <- NA
    }
  } else {
    for (j in seq_len(p))
      X_miss[sample(n, max(1L, round(rate * n)), replace = FALSE), j] <- NA
  }
  X_miss
}

# ---- 4b. Classical baselines (train missing only; test complete) ----

# MeanSI — fill train NAs with training means; test complete → unchanged
impute_mean <- function(X_tr_miss, X_te_full) {
  fill_vals <- sapply(X_tr_miss, function(col) {
    fv <- mean(col, na.rm = TRUE); if (is.na(fv)) 0 else fv
  })
  fill_df <- function(df) {
    out <- as.data.frame(df)
    for (nm in names(out)) {
      if (!nm %in% names(fill_vals)) next
      idx <- is.na(out[[nm]]); if (!any(idx)) next
      out[[nm]][idx] <- fill_vals[[nm]]
    }
    out
  }
  list(train = fill_df(X_tr_miss),
       test  = as.data.frame(X_te_full),
       note  = "train_means_only_test_complete",
       fallback = FALSE,
       fallback_type = NA_character_)
}

# RFI — missForest on TRAIN only; test complete → unchanged
impute_rfi <- function(X_tr_miss, X_te_full) {
  X_tr_imp <- tryCatch(
    suppressMessages(suppressWarnings(
      as.data.frame(
        missForest::missForest(as.data.frame(X_tr_miss), verbose = FALSE)$ximp))),
    error = function(e) NULL)
  if (is.null(X_tr_imp)) {
    out <- impute_mean(X_tr_miss, X_te_full)
    out$note <- "RFI_failed_fallback_MeanSI"
    out$fallback <- TRUE
    out$fallback_type <- "RFI_failed_fallback_MeanSI"
    return(out)
  }
  list(train = X_tr_imp,
       test  = as.data.frame(X_te_full),
       note  = "missForest_train_only_test_complete",
       fallback = FALSE,
       fallback_type = NA_character_)
}

# MissInd — mean fill + indicators on TRAIN; test indicators all zero
impute_missind <- function(X_tr_miss, X_te_full) {
  col_means <- sapply(X_tr_miss, function(col) {
    mv <- mean(col, na.rm = TRUE); if (is.na(mv)) 0 else mv
  })
  missing_cols <- names(which(sapply(X_tr_miss, function(col) any(is.na(col)))))
  
  train_out <- as.data.frame(X_tr_miss)
  for (nm in names(train_out)) {
    idx <- is.na(train_out[[nm]]); if (!any(idx)) next
    train_out[[nm]][idx] <- col_means[[nm]]
  }
  for (nm in missing_cols)
    train_out[[paste0(nm, "_ind")]] <- as.integer(is.na(X_tr_miss[[nm]]))
  
  test_out <- as.data.frame(X_te_full)
  for (nm in missing_cols)
    test_out[[paste0(nm, "_ind")]] <- 0L
  
  list(train = train_out, test = test_out,
       note = "mean_fill_plus_indicators_train_only",
       fallback = FALSE,
       fallback_type = NA_character_)
}

# CCA — fit on complete training cases; evaluate on ALL test rows (complete)
prepare_cca <- function(X_tr_miss, X_te_full, time_tr, event_tr) {
  cc_tr <- complete.cases(X_tr_miss)
  n_tr_cc <- sum(cc_tr)
  n_ev_cc <- sum(event_tr[cc_tr] == 1L, na.rm = TRUE)
  n_te <- nrow(X_te_full)
  list(
    train     = as.data.frame(X_tr_miss[cc_tr, , drop = FALSE]),
    test      = as.data.frame(X_te_full),
    time_tr   = time_tr[cc_tr],
    event_tr  = event_tr[cc_tr],
    te_keep   = rep(TRUE, n_te),
    usable    = (n_tr_cc >= 5L && n_ev_cc >= 2L && n_te >= 5L),
    note      = sprintf("CCA n_tr=%d n_te=%d (test_complete)", n_tr_cc, n_te)
  )
}

# ---- 5. MICE by imputation engine ----
# Nelson–Aalen cumulative hazard H(t) at each subject's observed time.
# Note: survival::survfit type="fh" is NOT a different hazard; it is
#   stype=2, ctype=1  →  S(t) = exp(-H_NA(t)).
# We use cumhaz directly (ctype=1) so the name matches White & Royston.
nelson_aalen <- function(time_vec, event_vec) {
  n <- length(time_vec)
  tryCatch({
    sf <- survival::survfit(
      survival::Surv(time_vec, event_vec) ~ 1,
      stype = 2, ctype = 1, se.fit = FALSE
    )
    # sf$cumhaz is the Nelson–Aalen estimator (same as -log(sf$surv) here)
    H <- as.numeric(sf$cumhaz)
    t_grid <- as.numeric(sf$time)
    vapply(seq_len(n), function(i) {
      ti <- time_vec[i]
      if (!is.finite(ti) || ti <= 0) return(0)
      idx <- which(t_grid <= ti)
      if (length(idx) == 0L) return(0)
      h <- H[max(idx)]
      if (!is.finite(h) || h < 0) 0 else h
    }, numeric(1))
  }, error = function(e) rep(0, n))
}

# MICE on TRAIN only; TEST is complete → replicate the same complete matrix m times.
# Survival-aware MI: Nelson–Aalen cumulative hazard (.na_H) is a predictor of
# incomplete covariates (White & Royston). Raw .time / .event are never imputed.
mice_one_engine <- function(X_tr_miss, time_tr, event_tr,
                            X_te_full, engine, seed = 1,
                            m = M_IMP, maxit = MAXIT_MICE) {
  stopifnot(engine %in% IMP_ENGINES)
  X_cols <- names(X_tr_miss)
  te0 <- as.data.frame(X_te_full)[, X_cols, drop = FALSE]
  # Artificial protocol: test matrix must be complete
  if (anyNA(te0)) {
    stop("mice_one_engine: test matrix has NA — artificial protocol requires complete TEST.")
  }
  
  mean_fill_list <- function() {
    fill <- sapply(X_tr_miss, function(z) {
      mv <- mean(z, na.rm = TRUE); if (is.na(mv)) 0 else mv
    })
    tr0 <- as.data.frame(X_tr_miss)
    for (nm in X_cols) {
      idx <- is.na(tr0[[nm]]); if (!any(idx)) next
      tr0[[nm]][idx] <- fill[[nm]]
    }
    tr0 <- tr0[, X_cols, drop = FALSE]
    list(train = replicate(m, tr0, simplify = FALSE),
         test  = replicate(m, te0, simplify = FALSE),
         note  = "mice_failed_mean_fill_train_test_complete",
         fallback = TRUE,
         fallback_type = "mice_failed_mean_fill",
         used_engine = NA_character_)
  }
  
  na_H_tr <- nelson_aalen(time_tr, event_tr)
  df_tr <- data.frame(X_tr_miss,
                      .na_H = na_H_tr, .time = time_tr, .event = event_tr)
  meths_tr <- setNames(rep(engine, ncol(df_tr)), names(df_tr))
  meths_tr[c(".na_H", ".time", ".event")] <- ""
  for (nm in names(df_tr)) if (!any(is.na(df_tr[[nm]]))) meths_tr[nm] <- ""
  pm_tr <- mice::make.predictorMatrix(df_tr)
  pm_tr[c(".na_H", ".time", ".event"), ] <- 0L
  
  run_mice <- function(df, meths, pm, seed_i, m_i) {
    tryCatch(
      suppressMessages(suppressWarnings(
        mice::mice(df, m = m_i, method = meths, predictorMatrix = pm,
                   seed = seed_i, printFlag = FALSE, maxit = maxit))),
      error = function(e) NULL)
  }
  
  mids_tr <- run_mice(df_tr, meths_tr, pm_tr, seed, m)
  used_engine <- engine
  fallback_type <- NA_character_
  if (is.null(mids_tr)) {
    for (fb in setdiff(c("pmm", "cart", "norm"), engine)) {
      meths_fb <- meths_tr
      meths_fb[meths_fb == engine] <- fb
      mids_tr <- run_mice(df_tr, meths_fb, pm_tr, seed + 7L, m)
      if (!is.null(mids_tr)) {
        used_engine <- fb
        fallback_type <- paste0("mice_engine_swap_", engine, "_to_", fb)
        break
      }
    }
  }
  if (is.null(mids_tr)) {
    out <- mean_fill_list()
    out$fallback <- TRUE
    out$fallback_type <- "mice_failed_mean_fill"
    out$used_engine <- NA_character_
    return(out)
  }
  
  train_list <- lapply(seq_len(m), function(i)
    mice::complete(mids_tr, i)[, X_cols, drop = FALSE])
  test_list  <- replicate(m, te0, simplify = FALSE)
  
  list(train = train_list, test = test_list,
       note = if (is.na(fallback_type))
         sprintf("mice_%s_m=%d_train_only_test_complete", engine, m)
       else fallback_type,
       fallback = !is.na(fallback_type),
       fallback_type = fallback_type,
       used_engine = used_engine)
}

# Fit classifier on each of m imputations; average LP and S(t)
# Returns: lp_te, lp_tr (averaged), surv_mat (averaged), fit_last (for Cox native optional)
fit_mi_averaged <- function(clf, train_list, test_list, time_tr, event_tr,
                            times_grid) {
  m <- length(train_list)
  n_te <- nrow(test_list[[1]])
  n_tr <- length(time_tr)
  lp_te_mat <- matrix(NA_real_, n_te, m)
  lp_tr_mat <- matrix(NA_real_, n_tr, m)
  surv_arr  <- array(NA_real_, dim = c(n_te, length(times_grid), m))
  
  for (k in seq_len(m)) {
    params <- tune_surv_params(clf, train_list[[k]], time_tr, event_tr)
    pred <- safe_lp(clf, train_list[[k]], time_tr, event_tr, test_list[[k]],
                    params = params)
    lp_tr <- train_lp_from_fit(clf, pred$fit, train_list[[k]], time_tr, event_tr)
    S <- risk_to_surv(clf, pred$fit, test_list[[k]],
                      lp_tr, time_tr, event_tr, pred$lp, times_grid)
    lp_te_mat[, k] <- pred$lp
    lp_tr_mat[, k] <- lp_tr
    surv_arr[, , k] <- S
  }
  
  list(
    lp_te    = rowMeans(lp_te_mat, na.rm = TRUE),
    lp_tr    = rowMeans(lp_tr_mat, na.rm = TRUE),
    surv_mat = apply(surv_arr, c(1, 2), mean, na.rm = TRUE),
    m        = m
  )
}

# ---- 6. Classifier wrappers (fixed defendable defaults; no inner CV tuning) ----
default_surv_params <- function(model_name) {
  switch(model_name,
         GBM = list(n.trees = GBM_N_TREES, shrinkage = 0.05, interaction.depth = 3L,
                    bag.fraction = 0.50),
         SurvKNN = list(k = KNN_K, num_base_learners = KNN_N_BAG),
         list()
  )
}

# Numeric design for bnnSurvival (Mahalanobis distance)
as_bnn_frame <- function(X) {
  X <- as.data.frame(X, stringsAsFactors = FALSE)
  for (j in seq_along(X)) {
    if (is.factor(X[[j]]) || is.character(X[[j]])) {
      X[[j]] <- as.numeric(factor(X[[j]]))
    } else {
      X[[j]] <- as.numeric(X[[j]])
    }
  }
  X
}

# Scalar risk from bnnSurvival S(t) curves (package has NO native risk score;
# exports are predictions()/timepoints() only). r = -mean(log S) = mean
# cumulative hazard on the predicted grid; higher = higher risk (Cox-LP polarity).
# IBS still uses native S(t); this scalar is for C-index / stacking meta only.
knn_risk_from_S <- function(S) {
  if (is.null(dim(S))) S <- matrix(S, nrow = 1L)
  as.numeric(-rowMeans(log(pmax(S, 1e-6)), na.rm = TRUE))
}

# Interpolate bnnSurvival S(t) (n × T) onto the evaluation time grid
knn_surv_to_grid <- function(S, s_times, times_grid) {
  n <- nrow(S)
  out <- matrix(NA_real_, n, length(times_grid))
  if (n < 1L || length(s_times) < 1L) return(out)
  for (j in seq_along(times_grid)) {
    idx <- max(which(s_times <= times_grid[j]), 0L)
    out[, j] <- if (idx == 0L) 1 else as.numeric(S[, idx])
  }
  out
}

# Alias kept so call sites stay simple (no tuning; returns fixed defaults).
tune_surv_params <- function(model_name, X_tr, time_tr, event_tr) {
  default_surv_params(model_name)
}

fit_surv_lp <- function(model_name, X_tr, time_tr, event_tr, X_te,
                        params = NULL) {
  p_names <- names(X_tr)
  df_tr   <- data.frame(X_tr, time = time_tr, event = event_tr)
  df_te   <- as.data.frame(X_te)[, p_names, drop = FALSE]
  n_te    <- nrow(df_te)
  f_surv  <- as.formula(paste("Surv(time, event) ~",
                              paste(p_names, collapse = " + ")))
  if (is.null(params)) {
    params <- tune_surv_params(model_name, X_tr, time_tr, event_tr)
  }
  
  suppressWarnings(tryCatch({
    if (model_name == "CoxPH") {
      m <- survival::coxph(f_surv, data = df_tr, ties = "efron", x = TRUE,
                           control = coxph.control(iter.max = 150, toler.inf = 0.75))
      attr(m, "surv_params") <- list()
      list(lp = as.numeric(predict(m, newdata = df_te, type = "lp")), fit = m)
    } else if (model_name == "GBM") {
      n_trees_fit <- max(as.integer(params$n.trees), 50L)
      m <- gbm::gbm(f_surv, data = df_tr, distribution = "coxph",
                    n.trees = n_trees_fit,
                    shrinkage = params$shrinkage,
                    interaction.depth = as.integer(params$interaction.depth),
                    bag.fraction = params$bag.fraction,
                    verbose = FALSE)
      best_iter <- as.integer(params$n.trees)
      attr(m, "surv_params") <- params
      attr(m, "best_iter") <- best_iter
      list(lp = as.numeric(gbm::predict.gbm(m, newdata = df_te,
                                            n.trees = best_iter)),
           fit = m)
    } else if (model_name == "SurvKNN") {
      # Bagged k-NN survival (Lowsky 2013; Wright bnnSurvival)
      Xtr_n <- as_bnn_frame(X_tr)
      Xte_n <- as_bnn_frame(df_te)
      k_use <- max(1L, min(as.integer(params$k), nrow(Xtr_n) - 1L))
      n_bag <- max(1L, as.integer(params$num_base_learners))
      df_fit <- data.frame(Xtr_n, time = time_tr, event = event_tr)
      f_knn <- as.formula(paste("Surv(time, event) ~",
                                paste(names(Xtr_n), collapse = " + ")))
      m <- bnnSurvival::bnnSurvival(
        f_knn, df_fit, k = k_use, num_base_learners = n_bag
      )
      attr(m, "surv_params") <- list(k = k_use, num_base_learners = n_bag)
      pr <- predict(m, Xte_n)
      S  <- bnnSurvival::predictions(pr)
      list(lp = knn_risk_from_S(S), fit = m)
    } else {
      list(lp = rep(NA_real_, n_te), fit = NULL)
    }
  }, error = function(e) list(lp = rep(NA_real_, n_te), fit = NULL)))
}

safe_lp <- function(model_name, X_tr, time_tr, event_tr, X_te, params = NULL) {
  n_te <- nrow(as.data.frame(X_te))
  n_tr_ok <- sum(!is.na(time_tr) & !is.na(event_tr))
  n_ev_ok <- sum(event_tr == 1L, na.rm = TRUE)
  if (n_tr_ok < max(ncol(as.data.frame(X_tr)) + 2L, 5L) || n_ev_ok < 2L)
    return(list(lp = rep(NA_real_, n_te), fit = NULL, params = NULL))
  out <- tryCatch(
    fit_surv_lp(model_name, X_tr, time_tr, event_tr, X_te, params = params),
    error = function(e) list(lp = rep(NA_real_, n_te), fit = NULL))
  if (length(out$lp) != n_te) out$lp <- rep(NA_real_, n_te)
  if (is.null(out$params)) {
    out$params <- if (!is.null(out$fit)) attr(out$fit, "surv_params") else params
  }
  out
}

# Align survfit S(t) matrix to a fixed time grid
# survival::survfit with newdata: sf$surv is (n_times x n_subjects).
survfit_to_grid <- function(fit, newdata, times_grid) {
  n <- nrow(newdata)
  empty <- matrix(NA_real_, n, length(times_grid))
  tryCatch({
    sf <- survival::survfit(fit, newdata = newdata, se.fit = FALSE)
    s_times <- sf$time
    if (is.null(dim(sf$surv))) {
      # Single curve — only valid if n == 1; else refuse (no silent recycle)
      if (n != 1L) return(empty)
      S <- matrix(sf$surv, ncol = 1L)
    } else {
      S <- sf$surv
      if (nrow(S) != length(s_times) || ncol(S) != n) return(empty)
    }
    out <- matrix(1, n, length(times_grid))
    for (j in seq_along(times_grid)) {
      idx <- max(which(s_times <= times_grid[j]), 0L)
      out[, j] <- if (idx == 0L) 1 else as.numeric(S[idx, ])
    }
    pmin(pmax(out, 0), 1)
  }, error = function(e) empty)
}

# Cox-calibrate any risk score → S(t)
# score_tr used to fit Surv ~ score; score_te mapped to S(t)
calibrate_surv <- function(score_tr, time_tr, event_tr, score_te, times_grid) {
  n_te <- length(score_te)
  empty <- matrix(NA_real_, n_te, length(times_grid))
  df_tr <- data.frame(score = as.numeric(score_tr),
                      time = time_tr, event = event_tr)
  df_tr <- df_tr[is.finite(df_tr$score) & is.finite(df_tr$time), , drop = FALSE]
  if (nrow(df_tr) < 5L || sum(df_tr$event == 1L) < 2L) return(empty)
  
  fit <- tryCatch(
    survival::coxph(Surv(time, event) ~ score, data = df_tr, ties = "efron",
                    x = TRUE),
    error = function(e) NULL)
  if (is.null(fit)) return(empty)
  
  df_te <- data.frame(score = as.numeric(score_te))
  df_te$score[!is.finite(df_te$score)] <- median(df_tr$score, na.rm = TRUE)
  survfit_to_grid(fit, df_te, times_grid)
}

# Native Cox / SurvKNN S(t) when available; else Cox-calibrate risk score
risk_to_surv <- function(model_name, fit, X_te, lp_tr, time_tr, event_tr,
                         lp_te, times_grid) {
  if (model_name == "CoxPH" && inherits(fit, "coxph")) {
    p_names <- names(X_te)
    df_te <- as.data.frame(X_te)[, p_names, drop = FALSE]
    S <- survfit_to_grid(fit, df_te, times_grid)
    if (all(is.na(S))) {
      return(calibrate_surv(lp_tr, time_tr, event_tr, lp_te, times_grid))
    }
    return(S)
  }
  if (model_name == "SurvKNN" && inherits(fit, "bnnSurvivalEnsemble")) {
    S_native <- tryCatch({
      pr <- predict(fit, as_bnn_frame(X_te))
      knn_surv_to_grid(bnnSurvival::predictions(pr),
                       bnnSurvival::timepoints(pr), times_grid)
    }, error = function(e) NULL)
    if (!is.null(S_native) && !all(is.na(S_native))) return(S_native)
  }
  calibrate_surv(lp_tr, time_tr, event_tr, lp_te, times_grid)
}

# Training LP (for calibration) — in-sample predict on training design
train_lp_from_fit <- function(model_name, fit, X_tr, time_tr, event_tr) {
  n <- nrow(X_tr)
  if (is.null(fit)) return(rep(NA_real_, n))
  tryCatch({
    if (model_name == "CoxPH" && inherits(fit, "coxph")) {
      as.numeric(predict(fit, type = "lp"))
    } else if (model_name == "GBM" && inherits(fit, "gbm")) {
      bi <- attr(fit, "best_iter")
      if (is.null(bi) || !is.finite(bi)) bi <- 200L
      as.numeric(gbm::predict.gbm(fit, newdata = data.frame(X_tr),
                                  n.trees = as.integer(bi)))
    } else if (model_name == "SurvKNN" && inherits(fit, "bnnSurvivalEnsemble")) {
      pr <- predict(fit, as_bnn_frame(X_tr))
      knn_risk_from_S(bnnSurvival::predictions(pr))
    } else rep(NA_real_, n)
  }, error = function(e) rep(NA_real_, n))
}

# CoxPH meta-learner stacking (NO Super Learner / NO ridge).
# Level-0 features = OOF LPs of PMM/RF/CART/NORM/MIDASTOUCH under fixed classifier.
# Meta = ALWAYS unpenalized CoxPH on those Level-0 LPs.
# S(t) via Cox-calibration of the meta linear predictor.
# Fallback if meta fit fails: equal average of Level-0 test LPs / S(t).
fit_meta_coxph <- function(cand_lp_tr, cand_lp_te, cand_surv_te,
                           time_tr, event_tr, times_grid,
                           library_names = NULL) {
  meta_model <- META_LEARNER  # "CoxPH"
  if (!is.null(library_names)) {
    keep <- intersect(library_names, names(cand_lp_tr))
    cand_lp_tr   <- cand_lp_tr[keep]
    cand_lp_te   <- cand_lp_te[keep]
    cand_surv_te <- cand_surv_te[keep]
  }
  nms <- names(cand_lp_tr)
  M <- length(nms)
  empty <- list(lp = NULL, lp_tr = NULL, surv = NULL,
                chosen = NA_character_, oof_c = NA_real_,
                weights = NULL, meta_coefs_str = NA_character_,
                meta_clf = meta_model)
  
  if (M < 1L) return(empty)
  
  n_tr <- length(time_tr)
  Z <- matrix(NA_real_, n_tr, M, dimnames = list(NULL, nms))
  for (j in seq_len(M)) {
    lp <- as.numeric(cand_lp_tr[[nms[j]]])
    if (length(lp) == n_tr) Z[, j] <- lp
  }
  ok_col <- colSums(is.finite(Z)) >= 5L
  if (!any(ok_col)) return(empty)
  if (!all(ok_col)) {
    nms <- nms[ok_col]
    Z <- Z[, ok_col, drop = FALSE]
    M <- length(nms)
    cand_lp_te   <- cand_lp_te[nms]
    cand_surv_te <- cand_surv_te[nms]
  }
  
  n_te <- length(as.numeric(cand_lp_te[[nms[1]]]))
  Z_te <- matrix(NA_real_, n_te, M, dimnames = list(NULL, nms))
  for (j in seq_len(M)) Z_te[, j] <- as.numeric(cand_lp_te[[nms[j]]])
  
  cn <- paste0("L0_", make.names(nms, unique = TRUE))
  colnames(Z) <- cn
  colnames(Z_te) <- cn
  
  # Impute residual NA in Level-0 features with column medians (train)
  for (j in seq_len(M)) {
    med <- median(Z[, j], na.rm = TRUE)
    if (!is.finite(med)) med <- 0
    Z[!is.finite(Z[, j]), j] <- med
    Z_te[!is.finite(Z_te[, j]), j] <- med
  }
  
  equal_fallback <- function() {
    w <- rep(1 / M, M); names(w) <- nms
    lp_te <- as.numeric(Z_te %*% w)
    lp_tr <- as.numeric(Z %*% w)
    S1 <- cand_surv_te[[nms[1]]]
    surv <- if (is.null(S1) || !is.matrix(S1)) {
      NULL
    } else {
      out <- matrix(0, nrow(S1), ncol(S1))
      for (j in seq_len(M)) {
        Sj <- cand_surv_te[[nms[j]]]
        if (!is.null(Sj) && is.matrix(Sj)) out <- out + w[j] * Sj
      }
      out
    }
    list(lp = lp_te, lp_tr = lp_tr, surv = surv,
         chosen = "equal_avg_fallback", oof_c = NA_real_,
         weights = w,
         meta_coefs_str = paste(sprintf("%s=%.4f", nms, w), collapse = ";"),
         meta_clf = paste0(meta_model, "_fallback_equal"))
  }
  
  n_ev <- sum(event_tr == 1L, na.rm = TRUE)
  if (n_ev < 3L || n_tr < M + 2L) return(equal_fallback())
  
  df_tr <- data.frame(Z, time = time_tr, event = event_tr, check.names = FALSE)
  df_te <- data.frame(Z_te, check.names = FALSE)
  f_surv <- as.formula(paste("Surv(time, event) ~", paste(cn, collapse = " + ")))
  fit_u <- tryCatch(
    suppressWarnings(
      survival::coxph(f_surv, data = df_tr, ties = "efron", x = TRUE)
    ),
    error = function(e) NULL
  )
  if (is.null(fit_u)) return(equal_fallback())
  
  lp_tr <- as.numeric(predict(fit_u, type = "lp"))
  lp_te <- as.numeric(predict(fit_u, newdata = df_te, type = "lp"))
  if (!any(is.finite(lp_te))) return(equal_fallback())
  
  S <- calibrate_surv(lp_tr, time_tr, event_tr, lp_te, times_grid)
  b <- tryCatch(coef(fit_u), error = function(e) NULL)
  coefs <- setNames(rep(NA_real_, M), nms)
  if (!is.null(b)) {
    for (j in seq_len(M)) {
      if (cn[j] %in% names(b)) coefs[j] <- unname(b[cn[j]])
    }
  }
  
  list(
    lp     = lp_te,
    lp_tr  = lp_tr,
    surv   = S,
    chosen = "meta_CoxPH",
    oof_c  = NA_real_,
    weights = coefs,
    meta_coefs_str = paste(sprintf("%s=%.4f", nms, coefs), collapse = ";"),
    meta_clf = meta_model
  )
}

# fit_meta_same_clf: thin wrapper kept for call-site clarity
fit_meta_same_clf <- function(clf, cand_lp_tr, cand_lp_te, cand_surv_te,
                              time_tr, event_tr, times_grid,
                              library_names = NULL) {
  fit_meta_coxph(cand_lp_tr, cand_lp_te, cand_surv_te,
                 time_tr, event_tr, times_grid,
                 library_names = library_names)
}

# ---- 7. OOF Level-0 scores (within one classifier) ----
# OOF LP on one completed training matrix (stacking Level-0). Fixed HPs.
get_oof_one_matrix <- function(clf, X_tr, time_tr, event_tr,
                               folds = NULL, cv_folds = CV_FOLDS,
                               params = NULL) {
  n <- length(time_tr)
  oof <- rep(NA_real_, n)
  X_tr <- as.data.frame(X_tr)
  if (is.null(folds)) {
    k_use <- min(cv_folds, sum(event_tr), n - 1L)
    if (k_use < 2L) return(oof)
    folds <- caret::createFolds(event_tr, k = k_use, list = TRUE,
                                returnTrain = FALSE)
  }
  if (is.null(params)) {
    params <- default_surv_params(clf)
  }
  for (fi in folds) {
    tri <- setdiff(seq_len(n), fi)
    pred <- safe_lp(clf, X_tr[tri, , drop = FALSE], time_tr[tri], event_tr[tri],
                    X_tr[fi, , drop = FALSE], params = params)
    oof[fi] <- pred$lp
  }
  oof
}

get_oof_by_engine <- function(clf, imp_list_tr, time_tr, event_tr,
                              engines = IMP_ENGINES, cv_folds = CV_FOLDS) {
  n <- length(time_tr)
  oof <- matrix(NA_real_, n, length(engines), dimnames = list(NULL, engines))
  k_use <- min(cv_folds, sum(event_tr), n - 1L)
  if (k_use < 2) return(oof)
  folds <- caret::createFolds(event_tr, k = k_use, list = TRUE, returnTrain = FALSE)
  
  for (eng in engines) {
    train_list <- imp_list_tr[[eng]]
    m <- length(train_list)
    oof_m <- matrix(NA_real_, n, m)
    for (k in seq_len(m)) {
      oof_m[, k] <- get_oof_one_matrix(clf, train_list[[k]], time_tr, event_tr,
                                       folds = folds)
    }
    oof[, eng] <- rowMeans(oof_m, na.rm = TRUE)
  }
  oof
}

# ImpStack: for each imputation copy k, stack engines then
# average the m stacked LP / S(t) predictions.
fit_impstack_per_copy_avg <- function(clf, imp_tr, imp_te, time_tr, event_tr,
                                      times_grid, engines = IMP_ENGINES) {
  m <- length(imp_tr[[engines[1]]])
  n_te <- nrow(imp_te[[engines[1]]][[1]])
  n_tr <- length(time_tr)
  n_t  <- length(times_grid)
  
  empty <- list(lp = NULL, surv = NULL, chosen = NA_character_,
                meta_coefs_str = NA_character_, meta_clf = META_LEARNER,
                oof_c = NA_real_)
  
  k_use <- min(CV_FOLDS, sum(event_tr), n_tr - 1L)
  if (k_use < 2L || m < 1L) return(empty)
  folds <- caret::createFolds(event_tr, k = k_use, list = TRUE,
                              returnTrain = FALSE)
  
  lp_te_m   <- matrix(NA_real_, n_te, m)
  surv_arr  <- array(NA_real_, dim = c(n_te, n_t, m))
  coef_mat  <- NULL
  n_meta_fallback <- 0L
  fallback_notes <- character(0)
  
  for (k in seq_len(m)) {
    cand_lp_tr <- list()
    cand_lp_te <- list()
    cand_surv  <- list()
    
    for (eng in engines) {
      nm <- unname(ENGINE_METHOD[[eng]])
      Xk_tr <- as.data.frame(imp_tr[[eng]][[k]])
      Xk_te <- as.data.frame(imp_te[[eng]][[k]])
      
      params <- tune_surv_params(clf, Xk_tr, time_tr, event_tr)
      oof_k <- get_oof_one_matrix(clf, Xk_tr, time_tr, event_tr,
                                  folds = folds, params = params)
      pred  <- safe_lp(clf, Xk_tr, time_tr, event_tr, Xk_te, params = params)
      lp_tr <- train_lp_from_fit(clf, pred$fit, Xk_tr, time_tr, event_tr)
      Sk    <- risk_to_surv(clf, pred$fit, Xk_te,
                            lp_tr, time_tr, event_tr, pred$lp, times_grid)
      
      cand_lp_tr[[nm]] <- oof_k
      cand_lp_te[[nm]] <- pred$lp
      cand_surv[[nm]]  <- Sk
    }
    
    meta_k <- fit_meta_coxph(
      cand_lp_tr, cand_lp_te, cand_surv,
      time_tr, event_tr, times_grid,
      library_names = unname(ENGINE_METHOD[engines]))
    
    if (is.null(meta_k$lp) || is.null(meta_k$surv)) next
    if (identical(meta_k$chosen, "equal_avg_fallback") ||
        grepl("fallback", as.character(meta_k$meta_clf), fixed = TRUE)) {
      n_meta_fallback <- n_meta_fallback + 1L
      fallback_notes <- c(fallback_notes, sprintf("copy%d_equal_avg_fallback", k))
    }
    lp_te_m[, k] <- meta_k$lp
    if (is.matrix(meta_k$surv) &&
        nrow(meta_k$surv) == n_te && ncol(meta_k$surv) == n_t) {
      surv_arr[, , k] <- meta_k$surv
    }
    if (!is.null(meta_k$weights)) {
      if (is.null(coef_mat)) {
        coef_mat <- matrix(NA_real_, m, length(meta_k$weights),
                           dimnames = list(NULL, names(meta_k$weights)))
      }
      coef_mat[k, names(meta_k$weights)] <- meta_k$weights
    }
  }
  
  if (!any(is.finite(lp_te_m))) return(empty)
  
  lp_avg <- rowMeans(lp_te_m, na.rm = TRUE)
  surv_avg <- apply(surv_arr, c(1, 2), mean, na.rm = TRUE)
  
  w_str <- NA_character_
  if (!is.null(coef_mat)) {
    w_mean <- colMeans(coef_mat, na.rm = TRUE)
    w_str <- paste(sprintf("%s=%.4f", names(w_mean), w_mean), collapse = ";")
  }
  
  list(
    lp = lp_avg,
    surv = surv_avg,
    chosen = paste0("per_copy_avg_m", m),
    meta_coefs_str = w_str,
    meta_clf = META_LEARNER,
    oof_c = NA_real_,
    n_copies = m,
    n_meta_fallback = n_meta_fallback,
    fallback = n_meta_fallback > 0L,
    fallback_type = if (n_meta_fallback > 0L)
      paste0("ImpStack_meta_equal_avg_fallback_x", n_meta_fallback)
    else NA_character_,
    fallback_notes = if (length(fallback_notes)) paste(fallback_notes, collapse = ";")
    else NA_character_
  )
}

# ---- 8. Metrics (paper definitions — keep in sync with METRICS.md) ----
# All risk scores are HIGHER = HIGHER HAZARD (worse prognosis):
#   CoxPH LP, GBM coxph score, SurvKNN -mean(log S), ImpStack meta Cox LP.
#
# Harrell C: survival::concordance(Surv ~ I(-risk)); higher better; in (0,1).
# CalSlope:  coef of coxph(Surv ~ risk) on TEST; target 1 (perfect Cox-scale).
# IBS:       Graf IPCW Brier integrated to tau = max(time grid); lower better.
#            S(t) from native Cox/SurvKNN or Cox-calibration of the risk score.
# MIE-SE:    C + CalSlope only (IBS NA: mixed Level-0 scales).

compute_cindex <- function(risk_scores, time_te, event_te) {
  risk  <- as.numeric(risk_scores)
  time  <- as.numeric(time_te)
  event <- as.integer(event_te)
  risk[is.infinite(risk)] <- NA_real_
  ok <- is.finite(risk) & is.finite(time) & time > 0 & event %in% c(0L, 1L)
  if (sum(ok) < 5L || sum(event[ok] == 1L) < 2L) return(NA_real_)
  # Larger -risk <=> lower hazard <=> longer expected survival (Harrell).
  tryCatch(
    survival::concordance(
      survival::Surv(time[ok], event[ok]) ~ I(-risk[ok]))$concordance,
    error = function(e) NA_real_)
}

# Test-set Cox calibration slope of the final risk score (target = 1).
# Same score polarity as C-index (higher = higher hazard).
# Stability: winsorize scores at 1%/99% (leverage control); |slope|>20 → NA.
compute_cal_slope <- function(risk_scores, time_te, event_te) {
  risk  <- as.numeric(risk_scores)
  time  <- as.numeric(time_te)
  event <- as.integer(event_te)
  risk[is.infinite(risk)] <- NA_real_
  ok <- is.finite(risk) & is.finite(time) & time > 0 & event %in% c(0L, 1L)
  if (sum(ok) < 5L || sum(event[ok] == 1L) < 2L) return(NA_real_)
  score <- risk[ok]
  qs <- stats::quantile(score, probs = c(0.01, 0.99), na.rm = TRUE, names = FALSE,
                        type = 7)
  if (all(is.finite(qs)) && qs[2] > qs[1]) {
    score <- pmin(pmax(score, qs[1]), qs[2])
  }
  df <- data.frame(time = time[ok], event = event[ok], score = score)
  if (stats::sd(df$score) < 1e-12) return(NA_real_)
  fit <- tryCatch(
    survival::coxph(survival::Surv(time, event) ~ score, data = df,
                    ties = "efron"),
    error = function(e) NULL)
  if (is.null(fit) || length(coef(fit)) < 1L) return(NA_real_)
  b <- as.numeric(coef(fit)[["score"]])
  if (!is.finite(b) || abs(b) > 20) return(NA_real_)  # numerically unstable
  b
}

# t-interval for a vector of dataset-level means (grand-mean CI)
mean_ci_lo <- function(x, conf = 0.95) {
  x <- x[is.finite(x)]; n <- length(x)
  if (n < 2L) return(NA_real_)
  m <- mean(x); se <- stats::sd(x) / sqrt(n)
  as.numeric(m - stats::qt(1 - (1 - conf) / 2, df = n - 1) * se)
}
mean_ci_hi <- function(x, conf = 0.95) {
  x <- x[is.finite(x)]; n <- length(x)
  if (n < 2L) return(NA_real_)
  m <- mean(x); se <- stats::sd(x) / sqrt(n)
  as.numeric(m + stats::qt(1 - (1 - conf) / 2, df = n - 1) * se)
}

# Graf et al. IPCW Brier score at grid times, trapezoidal IBS / tau.
# Status at t: event by t → y=0 weight 1/G(T_i); at risk past t → y=1 weight 1/G(t);
# censored before t → excluded. G = KM of censoring on the evaluation sample.
ipcw_ibs <- function(surv_mat, times, time_te, event_te) {
  time_te  <- as.numeric(time_te)
  event_te <- as.integer(event_te)
  n <- length(time_te)
  if (n < 5L || sum(event_te == 1L, na.rm = TRUE) < 2L) return(NA_real_)
  
  times <- as.numeric(times)
  ord <- order(times)
  times <- times[ord]
  if (length(times) < 2L || !all(is.finite(times))) return(NA_real_)
  
  if (is.null(dim(surv_mat))) surv_mat <- matrix(surv_mat, nrow = n)
  if (nrow(surv_mat) != n || ncol(surv_mat) != length(times)) {
    # Defensive: refuse misaligned S(t) rather than silently recycle
    return(NA_real_)
  }
  surv_mat <- surv_mat[, ord, drop = FALSE]
  surv_mat[!is.finite(surv_mat)] <- NA_real_
  surv_mat <- pmin(pmax(surv_mat, 0), 1)
  
  Gfit <- tryCatch(
    survival::survfit(survival::Surv(time_te, 1L - event_te) ~ 1),
    error = function(e) NULL)
  if (is.null(Gfit)) return(NA_real_)
  
  G_at <- function(t) {
    if (!is.finite(t) || t <= 0) return(1)
    idx <- max(which(Gfit$time <= t), 0L)
    if (idx == 0L) 1 else max(as.numeric(Gfit$surv[idx]), 1e-6)
  }
  
  brier_t <- rep(NA_real_, length(times))
  for (j in seq_along(times)) {
    t0 <- times[j]
    S_hat <- surv_mat[, j]
    w <- rep(0, n); y <- rep(NA_real_, n)
    for (i in seq_len(n)) {
      ti <- time_te[i]; di <- event_te[i]
      if (!is.finite(ti) || ti <= 0 || is.na(di) || !is.finite(S_hat[i])) next
      if (ti <= t0 && di == 1L) {
        w[i] <- 1 / G_at(ti); y[i] <- 0
      } else if (ti > t0) {
        w[i] <- 1 / G_at(t0); y[i] <- 1
      }
      # else: censored at ti <= t0 → excluded (IPCW)
    }
    ok <- is.finite(w) & is.finite(y) & is.finite(S_hat) & w > 0
    if (sum(ok) >= 2L)
      brier_t[j] <- sum(w[ok] * (y[ok] - S_hat[ok])^2) / sum(w[ok])
  }
  
  tau <- max(times)
  okb <- is.finite(brier_t)
  if (sum(okb) >= 2L && is.finite(tau) && tau > 0) {
    bt <- brier_t[okb]; tt <- times[okb]
    as.numeric(sum(diff(tt) * (head(bt, -1) + tail(bt, -1)) / 2) / tau)
  } else NA_real_
}

make_time_grid <- function(time_tr, event_tr, time_te, n_grid = 20L) {
  ev <- time_tr[event_tr == 1L & is.finite(time_tr)]
  if (length(ev) < 4L) ev <- time_tr[is.finite(time_tr)]
  tau <- as.numeric(quantile(ev, 0.75, na.rm = TRUE))
  if (!is.finite(tau) || tau <= 0)
    tau <- as.numeric(quantile(time_te, 0.75, na.rm = TRUE))
  tmin <- max(min(ev[ev > 0], na.rm = TRUE), 1e-6)
  if (!is.finite(tmin) || tmin >= tau) tmin <- tau / n_grid
  seq(tmin, tau, length.out = n_grid)
}

# SurvProbSource documents how S(t) was obtained (for IBS reporting)
# MetaCoefs: ImpStack_MICE meta coefs/importances string (else NA)
# Fallback / FallbackType: report how often emergency paths fired (paper audit)
eval_one <- function(lp, surv_mat, times_grid, time_te, event_te,
                     method, classifier, surv_prob_source = NA_character_,
                     selected_learner = NA_character_,
                     oof_c_selected = NA_real_,
                     meta_coefs = NA_character_,
                     skip_ibs = FALSE,
                     fallback = FALSE,
                     fallback_type = NA_character_) {
  # skip_ibs: keep C/CalSlope; leave IBS NA (legacy flag; prefer Cox-calibrated S(t))
  data.frame(
    Classifier       = classifier,
    Method           = method,
    Cindex           = compute_cindex(lp, time_te, event_te),
    IBS              = if (isTRUE(skip_ibs)) NA_real_
    else ipcw_ibs(surv_mat, times_grid, time_te, event_te),
    CalSlope         = compute_cal_slope(lp, time_te, event_te),
    SurvProbSource   = surv_prob_source,
    SelectedLearner  = selected_learner,
    OOF_C_selected   = oof_c_selected,
    MetaCoefs       = meta_coefs,
    Fallback         = as.integer(isTRUE(fallback)),
    FallbackType     = if (isTRUE(fallback)) as.character(fallback_type) else NA_character_,
    n_eval           = length(time_te),
    stringsAsFactors = FALSE
  )
}

na_eval <- function(method, classifier, n_eval = 0L,
                    fallback = TRUE,
                    fallback_type = "unavailable_or_failed") {
  data.frame(
    Classifier = classifier, Method = method,
    Cindex = NA_real_,
    IBS = NA_real_, CalSlope = NA_real_,
    SurvProbSource = "unavailable",
    SelectedLearner = NA_character_,
    OOF_C_selected = NA_real_,
    MetaCoefs = NA_character_,
    Fallback = as.integer(isTRUE(fallback)),
    FallbackType = as.character(fallback_type),
    n_eval = as.integer(n_eval),
    stringsAsFactors = FALSE
  )
}

surv_source_label <- function(model_name) {
  if (identical(model_name, "CoxPH")) "native_coxph_survfit"
  else if (identical(model_name, "SurvKNN")) "native_bnnSurvival_knn"
  else "cox_calibration_of_risk_score"
}

# ---- 8b. MICE_SE (Aleryani et al. 2020 — original concatenation style) ----
# Heterogeneous Level-0 = all SURV_NAMES × all MICE-pmm copies.
# Original MIE-SE: concatenate K*m score columns (no MI-average collapse).
# Classifier column in results = meta-learner identity.
# Metrics: Harrell C, CalSlope (from final meta LP).
# IBS intentionally skipped: L0 concatenates heterogeneous classifier scales,
# so Cox-calibrated S(t) / IPCW-IBS would not be a fair survival-probability metric.
mice_se_l0_surv <- function(imp_tr, time_tr, event_tr, imp_te,
                            clfs = SURV_NAMES, oof_folds = CV_FOLDS) {
  m_use <- length(imp_tr)
  n_tr  <- length(time_tr)
  n_te  <- nrow(as.data.frame(imp_te[[1]]))
  if (m_use < 1L || n_tr < 5L || n_te < 1L)
    return(list(L0_tr = NULL, L0_te = NULL))
  
  k_use <- min(oof_folds, sum(event_tr == 1L, na.rm = TRUE), n_tr - 1L)
  if (k_use < 2L) return(list(L0_tr = NULL, L0_te = NULL))
  folds <- caret::createFolds(event_tr, k = k_use, list = TRUE,
                              returnTrain = FALSE)
  
  col_names <- as.vector(outer(clfs, seq_len(m_use),
                               function(sn, k) paste0(sn, "_m", k)))
  L0_tr <- matrix(NA_real_, n_tr, length(col_names),
                  dimnames = list(NULL, col_names))
  L0_te <- matrix(NA_real_, n_te, length(col_names),
                  dimnames = list(NULL, col_names))
  
  for (sn in clfs) {
    params <- default_surv_params(sn)
    for (k in seq_len(m_use)) {
      cn <- paste0(sn, "_m", k)
      Xk_tr <- as.data.frame(imp_tr[[k]])
      Xk_te <- as.data.frame(imp_te[[k]])
      L0_tr[, cn] <- get_oof_one_matrix(sn, Xk_tr, time_tr, event_tr,
                                        folds = folds, params = params)
      pred <- safe_lp(sn, Xk_tr, time_tr, event_tr, Xk_te, params = params)
      te_k <- as.numeric(pred$lp)
      if (length(te_k) == n_te) L0_te[, cn] <- te_k
    }
  }
  
  keep <- colSums(is.finite(L0_tr)) >= 5L & colSums(is.finite(L0_te)) >= 1L
  if (!any(keep)) return(list(L0_tr = NULL, L0_te = NULL))
  list(L0_tr = L0_tr[, keep, drop = FALSE],
       L0_te = L0_te[, keep, drop = FALSE],
       n_l0_cols = sum(keep))
}

# Returns list(lp_tr, lp_te). CalSlope uses lp_te; IBS is not evaluated for MIE-SE.
run_mice_se_surv <- function(L0_tr, L0_te, time_tr, event_tr, meta_surv) {
  n_te <- if (is.null(L0_te)) 0L else nrow(as.data.frame(L0_te))
  n_tr <- if (is.null(L0_tr)) 0L else nrow(as.data.frame(L0_tr))
  empty <- list(lp_tr = rep(NA_real_, n_tr), lp_te = rep(NA_real_, n_te))
  if (is.null(L0_tr) || is.null(L0_te)) return(empty)
  df_tr <- as.data.frame(L0_tr)
  df_te <- as.data.frame(L0_te)
  # fill rare NAs with column medians
  for (nm in names(df_tr)) {
    med <- stats::median(df_tr[[nm]], na.rm = TRUE)
    if (!is.finite(med)) med <- 0
    df_tr[[nm]][!is.finite(df_tr[[nm]])] <- med
    df_te[[nm]][!is.finite(df_te[[nm]])] <- med
  }
  fb_te <- as.numeric(rowMeans(as.matrix(df_te), na.rm = TRUE))
  fb_tr <- as.numeric(rowMeans(as.matrix(df_tr), na.rm = TRUE))
  
  tryCatch({
    if (identical(meta_surv, "CoxPH")) {
      df_tr$time <- time_tr; df_tr$event <- event_tr
      lp_terms <- paste(setdiff(names(df_tr), c("time", "event")), collapse = " + ")
      f_meta <- as.formula(paste("Surv(time, event) ~", lp_terms))
      m <- survival::coxph(f_meta, data = df_tr, ties = "efron",
                           control = coxph.control(iter.max = 150, toler.inf = 0.75))
      list(
        lp_tr = as.numeric(predict(m, newdata = df_tr, type = "lp")),
        lp_te = as.numeric(predict(m, newdata = df_te, type = "lp"))
      )
    } else {
      params <- default_surv_params(meta_surv)
      pred_te <- safe_lp(meta_surv, df_tr, time_tr, event_tr, df_te, params = params)
      pred_tr <- safe_lp(meta_surv, df_tr, time_tr, event_tr, df_tr, params = params)
      lp_te <- if (length(pred_te$lp) == n_te && any(is.finite(pred_te$lp)))
        as.numeric(pred_te$lp) else fb_te
      lp_tr <- if (length(pred_tr$lp) == n_tr && any(is.finite(pred_tr$lp)))
        as.numeric(pred_tr$lp) else fb_tr
      list(lp_tr = lp_tr, lp_te = lp_te)
    }
  }, error = function(e) list(lp_tr = fb_tr, lp_te = fb_te))
}

# ---- 9. One split: all classifiers, within-classifier comparison ----
run_one_split <- function(df, miss_rate, mech, split_seed) {
  set.seed(split_seed)
  n <- nrow(df)
  idx_e1 <- which(df$event == 1L); idx_e0 <- which(df$event == 0L)
  n_tr_e1 <- max(1L, floor(TRAIN_FRAC * length(idx_e1)))
  n_tr_e0 <- max(1L, floor(TRAIN_FRAC * length(idx_e0)))
  tr_idx <- c(sample(idx_e1, n_tr_e1), sample(idx_e0, n_tr_e0))
  te_idx <- setdiff(seq_len(n), tr_idx)
  
  pred_cols <- setdiff(names(df), c("time", "event"))
  X_tr <- df[tr_idx, pred_cols, drop = FALSE]
  X_te <- df[te_idx, pred_cols, drop = FALSE]
  time_tr <- df$time[tr_idx]; event_tr <- df$event[tr_idx]
  time_te <- df$time[te_idx]; event_te <- df$event[te_idx]
  
  # NATURAL: train keeps observed NAs; TEST drops incomplete rows (complete-case
  # evaluation), because MICE/baselines assume a complete test design matrix.
  # Artificial: missingness on TRAIN only; TEST already complete.
  if (toupper(mech) == "NATURAL") {
    X_tr_miss <- as.data.frame(X_tr)
    cc_te <- stats::complete.cases(X_te)
    n_te_cc <- sum(cc_te)
    if (n_te_cc < 5L) {
      stop(sprintf(
        "NATURAL: only %d complete test cases after dropping predictor NAs (need >= 5).",
        n_te_cc))
    }
    X_te_full <- as.data.frame(X_te[cc_te, , drop = FALSE])
    time_te   <- time_te[cc_te]
    event_te  <- event_te[cc_te]
    # optional diagnostics (picked up by timing attr below if present)
    attr(X_te_full, "n_test_before_cc") <- nrow(X_te)
    attr(X_te_full, "n_test_after_cc")  <- n_te_cc
  } else {
    X_tr_miss <- induce_missing(X_tr, miss_rate, mech, time_tr, event_tr,
                                seed = split_seed + 1L)
    X_te_full <- as.data.frame(X_te)
    if (anyNA(X_te_full)) {
      stop("Artificial protocol violated: TEST covariates must be complete.")
    }
  }
  
  times_grid <- make_time_grid(time_tr, event_tr, time_te)
  t_split0 <- proc.time()[3]
  t_mice0 <- proc.time()[3]
  
  # MICE engines: impute train only; test is complete
  imp_tr <- list(); imp_te <- list(); mice_fb <- list()
  for (j in seq_along(IMP_ENGINES)) {
    eng <- IMP_ENGINES[j]
    out <- mice_one_engine(X_tr_miss, time_tr, event_tr, X_te_full,
                           engine = eng, seed = split_seed + 100L * j,
                           m = M_IMP)
    imp_tr[[eng]] <- out$train
    imp_te[[eng]] <- out$test
    mice_fb[[eng]] <- list(
      fallback = isTRUE(out$fallback),
      fallback_type = if (!is.null(out$fallback_type)) out$fallback_type else NA_character_
    )
  }
  sec_mice <- proc.time()[3] - t_mice0
  t_within0 <- proc.time()[3]
  
  # Classical baselines (train missing; test complete)
  si_mean <- impute_mean(X_tr_miss, X_te_full)
  si_rfi  <- impute_rfi(X_tr_miss, X_te_full)
  si_mind <- tryCatch(
    impute_missind(X_tr_miss, X_te_full),
    error = function(e) {
      out <- impute_mean(X_tr_miss, X_te_full)
      out$note <- "MissInd_failed_fallback_MeanSI"
      out$fallback <- TRUE
      out$fallback_type <- "MissInd_failed_fallback_MeanSI"
      out
    })
  cca_obj <- prepare_cca(X_tr_miss, X_te_full, time_tr, event_tr)
  
  rows <- list()
  
  # ---- WITHIN each classifier ----
  for (clf in SURV_NAMES) {
    src <- surv_source_label(clf)
    
    # ---- MeanSI ----
    params <- tune_surv_params(clf, si_mean$train, time_tr, event_tr)
    pred <- safe_lp(clf, si_mean$train, time_tr, event_tr, si_mean$test,
                    params = params)
    lp_tr_si <- train_lp_from_fit(clf, pred$fit, si_mean$train, time_tr, event_tr)
    S <- risk_to_surv(clf, pred$fit, si_mean$test,
                      lp_tr_si, time_tr, event_tr, pred$lp, times_grid)
    rows[[length(rows) + 1L]] <- eval_one(
      pred$lp, S, times_grid, time_te, event_te,
      method = "MeanSI", classifier = clf,
      surv_prob_source = paste0("fixed_", src))
    
    # ---- RFI ----
    params <- tune_surv_params(clf, si_rfi$train, time_tr, event_tr)
    pred <- safe_lp(clf, si_rfi$train, time_tr, event_tr, si_rfi$test,
                    params = params)
    lp_tr_si <- train_lp_from_fit(clf, pred$fit, si_rfi$train, time_tr, event_tr)
    S <- risk_to_surv(clf, pred$fit, si_rfi$test,
                      lp_tr_si, time_tr, event_tr, pred$lp, times_grid)
    rows[[length(rows) + 1L]] <- eval_one(
      pred$lp, S, times_grid, time_te, event_te,
      method = "RFI", classifier = clf,
      surv_prob_source = paste0("fixed_", src),
      fallback = isTRUE(si_rfi$fallback),
      fallback_type = si_rfi$fallback_type)
    
    # ---- MissInd ----
    params <- tune_surv_params(clf, si_mind$train, time_tr, event_tr)
    pred <- safe_lp(clf, si_mind$train, time_tr, event_tr, si_mind$test,
                    params = params)
    lp_tr_si <- train_lp_from_fit(clf, pred$fit, si_mind$train, time_tr, event_tr)
    S <- risk_to_surv(clf, pred$fit, si_mind$test,
                      lp_tr_si, time_tr, event_tr, pred$lp, times_grid)
    rows[[length(rows) + 1L]] <- eval_one(
      pred$lp, S, times_grid, time_te, event_te,
      method = "MissInd", classifier = clf,
      surv_prob_source = paste0("fixed_", src),
      fallback = isTRUE(si_mind$fallback),
      fallback_type = si_mind$fallback_type)
    
    # ---- CCA ----
    if (!cca_obj$usable ||
        sum(event_te[cca_obj$te_keep] == 1L, na.rm = TRUE) < 2L) {
      rows[[length(rows) + 1L]] <- na_eval(
        "CCA", clf, n_eval = sum(cca_obj$te_keep),
        fallback = TRUE, fallback_type = "CCA_unusable_too_few_complete_cases")
    } else {
      params <- tune_surv_params(clf, cca_obj$train, cca_obj$time_tr,
                                 cca_obj$event_tr)
      pred <- safe_lp(clf, cca_obj$train, cca_obj$time_tr, cca_obj$event_tr,
                      cca_obj$test, params = params)
      lp_tr_si <- train_lp_from_fit(clf, pred$fit, cca_obj$train,
                                    cca_obj$time_tr, cca_obj$event_tr)
      tg_cca <- make_time_grid(cca_obj$time_tr, cca_obj$event_tr,
                               time_te[cca_obj$te_keep])
      S <- risk_to_surv(clf, pred$fit, cca_obj$test,
                        lp_tr_si, cca_obj$time_tr, cca_obj$event_tr,
                        pred$lp, tg_cca)
      rows[[length(rows) + 1L]] <- eval_one(
        pred$lp, S, tg_cca,
        time_te[cca_obj$te_keep], event_te[cca_obj$te_keep],
        method = "CCA", classifier = clf,
        surv_prob_source = paste0("fixed_", src))
    }
    
    # ---- MICE engines (m=5 each; average predictions within engine) ----
    for (eng in IMP_ENGINES) {
      avg <- fit_mi_averaged(clf, imp_tr[[eng]], imp_te[[eng]],
                             time_tr, event_tr, times_grid)
      nm <- unname(ENGINE_METHOD[[eng]])
      fb <- mice_fb[[eng]]
      rows[[length(rows) + 1L]] <- eval_one(
        avg$lp_te, avg$surv_mat, times_grid, time_te, event_te,
        method = nm, classifier = clf,
        surv_prob_source = paste0("mi_avg_m", M_IMP, "_fixed_", src),
        fallback = isTRUE(fb$fallback),
        fallback_type = fb$fallback_type)
    }
    
    # ---- ImpStack_MICE: per-copy stack then average ----
    meta_m <- fit_impstack_per_copy_avg(
      clf, imp_tr, imp_te, time_tr, event_tr, times_grid,
      engines = IMP_ENGINES)
    if (is.null(meta_m$lp) || is.null(meta_m$surv)) {
      rows[[length(rows) + 1L]] <- na_eval(
        "ImpStack_MICE", clf, n_eval = length(te_idx),
        fallback = TRUE, fallback_type = "ImpStack_failed_unavailable")
    } else {
      # Also flag if any Level-0 MICE engine used a fallback path
      eng_fb <- any(vapply(mice_fb, function(z) isTRUE(z$fallback), logical(1)))
      fb_types <- unique(na.omit(c(
        meta_m$fallback_type,
        if (eng_fb) {
          paste(vapply(IMP_ENGINES, function(e) {
            if (isTRUE(mice_fb[[e]]$fallback)) mice_fb[[e]]$fallback_type else NA_character_
          }, character(1)), collapse = ";")
        } else NA_character_
      )))
      fb_types <- fb_types[nzchar(fb_types) & !is.na(fb_types)]
      rows[[length(rows) + 1L]] <- eval_one(
        meta_m$lp, meta_m$surv, times_grid, time_te, event_te,
        method = "ImpStack_MICE", classifier = clf,
        surv_prob_source = paste0("perCopyStack_avg_m", M_IMP, "_fixed_meta_",
                                  meta_m$meta_clf),
        selected_learner = meta_m$chosen,
        oof_c_selected   = meta_m$oof_c,
        meta_coefs       = meta_m$meta_coefs_str,
        fallback = isTRUE(meta_m$fallback) || eng_fb,
        fallback_type = if (length(fb_types)) paste(fb_types, collapse = "|")
        else NA_character_)
    }
  }
  sec_within <- proc.time()[3] - t_within0
  
  # ---- MICE_SE / MIE-SE (cross-classifier; once per split) ----
  # Reuse MICE-pmm copies (Aleryani original Km columns). Classifier = meta.
  # Rank metrics: Harrell C, CalSlope (final meta LP).
  # IBS skipped: heterogeneous Level-0 scales → no coherent S(t) for IPCW-IBS.
  t_se0 <- proc.time()[3]
  imp_se_tr <- imp_tr[["pmm"]]
  imp_se_te <- imp_te[["pmm"]]
  n_l0_se <- NA_integer_
  pmm_fb <- mice_fb[["pmm"]]
  if (!is.null(imp_se_tr) && length(imp_se_tr) >= 1L &&
      !is.null(imp_se_te) && length(imp_se_te) >= 1L) {
    l0_se <- tryCatch(
      mice_se_l0_surv(imp_se_tr, time_tr, event_tr, imp_se_te),
      error = function(e) list(L0_tr = NULL, L0_te = NULL))
    if (!is.null(l0_se$L0_tr)) n_l0_se <- ncol(l0_se$L0_tr)
    for (meta_clf in SURV_NAMES) {
      if (is.null(l0_se$L0_tr) || is.null(l0_se$L0_te)) {
        rows[[length(rows) + 1L]] <- na_eval(
          "MICE_SE", meta_clf, n_eval = length(te_idx),
          fallback = TRUE, fallback_type = "MICE_SE_L0_failed")
      } else {
        se_fb <- FALSE
        se_fb_type <- NA_character_
        se_lp <- tryCatch(
          run_mice_se_surv(l0_se$L0_tr, l0_se$L0_te, time_tr, event_tr,
                           meta_clf),
          error = function(e) {
            se_fb <<- TRUE
            se_fb_type <<- "MICE_SE_meta_failed_rowMeans"
            list(
              lp_tr = as.numeric(rowMeans(l0_se$L0_tr, na.rm = TRUE)),
              lp_te = as.numeric(rowMeans(l0_se$L0_te, na.rm = TRUE))
            )
          })
        # Backward-compatible if an older helper still returns a bare vector
        if (!is.list(se_lp)) {
          se_fb <- TRUE
          se_fb_type <- "MICE_SE_meta_failed_rowMeans"
          se_lp <- list(
            lp_tr = as.numeric(rowMeans(l0_se$L0_tr, na.rm = TRUE)),
            lp_te = as.numeric(se_lp)
          )
        }
        fb_all <- isTRUE(se_fb) || isTRUE(pmm_fb$fallback)
        fb_type <- paste(na.omit(c(
          if (isTRUE(pmm_fb$fallback)) pmm_fb$fallback_type else NA_character_,
          se_fb_type
        )), collapse = "|")
        if (!nzchar(fb_type)) fb_type <- NA_character_
        # Placeholder S(t); IBS not used (skip_ibs = TRUE)
        S_na <- matrix(NA_real_, length(se_lp$lp_te), length(times_grid))
        rows[[length(rows) + 1L]] <- eval_one(
          se_lp$lp_te, S_na, times_grid, time_te, event_te,
          method = "MICE_SE", classifier = meta_clf,
          surv_prob_source = "ibs_not_applicable_mixed_L0_scales_MICE_SE",
          selected_learner = paste0("meta_", meta_clf),
          skip_ibs = TRUE,
          fallback = fb_all,
          fallback_type = fb_type)
      }
    }
  } else {
    for (meta_clf in SURV_NAMES) {
      rows[[length(rows) + 1L]] <- na_eval(
        "MICE_SE", meta_clf, n_eval = length(te_idx),
        fallback = TRUE, fallback_type = "MICE_SE_no_pmm_copies")
    }
  }
  sec_mice_se <- proc.time()[3] - t_se0
  sec_total <- proc.time()[3] - t_split0
  
  out <- dplyr::bind_rows(rows)
  n_test_eval <- length(time_te)
  n_test_raw <- length(te_idx)
  attr(out, "timing") <- data.frame(
    n_train = length(tr_idx),
    n_test = n_test_eval,                 # after NATURAL complete-case drop
    n_test_before_cc = n_test_raw,        # raw test size before NA drop
    n_pred = length(pred_cols),
    n_result_rows = nrow(out),
    mice_se_l0_cols = as.integer(n_l0_se),
    sec_mice = as.numeric(sec_mice),
    sec_within_clf = as.numeric(sec_within),
    sec_mice_se = as.numeric(sec_mice_se),
    sec_total = as.numeric(sec_total),
    stringsAsFactors = FALSE
  )
  out
}

# ---- 10. Main loop ----
COMP_STRUCT <- build_computation_structure()
write_computation_structure(COMP_STRUCT)
cat("Computation structure saved:",
    file.path(OUTPUT_DIR, "computation_structure.txt"), "\n")

cat("============================================================\n")
cat(" Imputation stacking — WITHIN each classifier\n")
cat(" Classifiers :", paste(SURV_NAMES, collapse = " | "), "\n")
cat(" Baselines   :", paste(SI_BASELINES, collapse = ", "), "\n")
cat(" Engines     :", paste(IMP_ENGINES, collapse = ", "),
    sprintf("(each MICE m=%d, predictions averaged)", M_IMP), "\n")
cat(" Stacking    : ImpStack_MICE = per-copy CoxPH meta on PMM/RF/CART/NORM/MIDASTOUCH, then avg m stacks\n")
cat(" Competitor  : MICE_SE / MIE-SE (Aleryani) = cross-clf stack; L0 = K*m; C/Cal (IBS skipped)\n")
cat(" Meta-learner:", META_LEARNER, "(unpenalized CoxPH; ImpStack only)\n")
cat(" MICE library:", paste(STACK_MICE_LIB, collapse = ", "), "\n")
cat(" Tuning      : fixed literature defaults (no inner CV for Level-0 classifiers)\n")
cat(" Train/Test  :", sprintf("%.0f/%.0f", 100 * TRAIN_FRAC, 100 * (1 - TRAIN_FRAC)), "\n")
cat(" Missingness : Artificial MAR/MCAR on TRAIN only; TEST stays complete\n")
cat(" Datasets    :", paste(vapply(ALL_DATASETS, `[[`, "", "name"), collapse = ", "),
    " [ARTIFICIAL ONLY]\n")
cat(" Per clf     :", paste(METHOD_ORDER, collapse = " | "), "\n")
cat(" Metrics     : Harrell C | IBS | CalSlope  (MIE-SE: no IBS; mixed L0 scales)\n")
cat(" Timing dir  :", TIMING_DIR, "\n")
cat(" Repro dir   :", REPRO_DIR, "\n")
cat("============================================================\n")

all_results <- list()
timing_split_log <- list()
timing_dataset_log <- list()
t0_run <- proc.time()[3]

append_split_timing <- function(ds_name, mech, rate, sp, res, status = "ok") {
  tm <- attr(res, "timing")
  if (is.null(tm) || !is.data.frame(tm) || nrow(tm) < 1L) {
    tm <- data.frame(
      n_train = NA_integer_, n_test = NA_integer_, n_pred = NA_integer_,
      n_result_rows = if (!is.null(res)) nrow(res) else NA_integer_,
      mice_se_l0_cols = NA_integer_,
      sec_mice = NA_real_, sec_within_clf = NA_real_,
      sec_mice_se = NA_real_, sec_total = NA_real_
    )
  }
  row <- data.frame(
    Dataset = ds_name, MissMech = mech, MissRate = rate, Split = sp,
    status = status, tm, stringsAsFactors = FALSE
  )
  timing_split_log[[length(timing_split_log) + 1L]] <<- row
  # flush incrementally so a kill still leaves timing evidence
  if (length(timing_split_log) %% 5L == 0L) {
    write_csv(dplyr::bind_rows(timing_split_log),
              file.path(TIMING_DIR, "timing_per_split.csv"))
  }
}

for (ds in ALL_DATASETS) {
  ds_name <- ds$name
  ds_mode <- if (!is.null(ds$mode)) ds$mode else "artificial"
  ckpt_file <- file.path(CHECKPOINT_DIR, paste0(ds_name, ".rds"))
  if (file.exists(ckpt_file)) {
    cat(sprintf(">>> SKIP (checkpoint): %s\n", ds_name))
    all_results[[ds_name]] <- readRDS(ckpt_file)
    timing_dataset_log[[length(timing_dataset_log) + 1L]] <- data.frame(
      Dataset = ds_name, status = "checkpoint_skip",
      sec_total = NA_real_, n_splits_done = NA_integer_,
      stringsAsFactors = FALSE)
    next
  }
  
  df <- tryCatch(ds$loader(), error = function(e) {
    message("Failed to load ", ds_name, ": ", e$message); NULL
  })
  if (is.null(df)) next
  
  n <- nrow(df); p <- ncol(df) - 2L
  n_ev <- sum(df$event == 1L)
  obs_rate <- obs_missing_rate(df)
  cat(sprintf("\n>>> Processing: %s  mode=%s  n=%d p=%d events=%d (%.0f%%)  obsNA=%.1f%%\n",
              ds_name, ds_mode, n, p, n_ev, 100 * n_ev / n, 100 * obs_rate))
  
  ds_rows <- list()
  t0_ds <- proc.time()[3]
  n_ok_sp <- 0L
  
  if (identical(ds_mode, "natural")) {
    mech <- "NATURAL"
    rate <- obs_rate
    cat(sprintf("  [%s] observed=%.1f%% ", mech, 100 * rate))
    for (sp in seq_len(N_SPLITS)) {
      seed <- make_split_seed(sp, ds_name, rate, mech)
      res <- tryCatch(
        run_one_split(df, miss_rate = 0, mech = mech, split_seed = seed),
        error = function(e) {
          message(sprintf("\n  [ERR %s sp=%d] %s", mech, sp, e$message))
          NULL
        })
      if (is.null(res)) {
        cat("x")
        append_split_timing(ds_name, mech, rate, sp, NULL, status = "error")
        next
      }
      tm <- attr(res, "timing")
      res$Dataset  <- ds_name
      res$MissMech <- mech
      res$MissRate <- rate
      res$Split    <- sp
      ds_rows[[length(ds_rows) + 1L]] <- res
      append_split_timing(ds_name, mech, rate, sp, res, status = "ok")
      n_ok_sp <- n_ok_sp + 1L
      cat(sprintf(".%.0fs", if (!is.null(tm)) tm$sec_total else NA_real_))
      flush.console()
      saveRDS(dplyr::bind_rows(ds_rows),
              file.path(CHECKPOINT_DIR, paste0(ds_name, "_partial.rds")))
      writeLines(
        sprintf("%s NATURAL split %d/%d done at %s",
                ds_name, sp, N_SPLITS, format(Sys.time())),
        file.path(OUTPUT_DIR, "progress.txt"))
    }
    cat(" done\n"); flush.console()
  } else {
    for (mech in MISS_MECHS) {
      cat(sprintf("  [%s] ", mech))
      for (rate in MISS_RATES) {
        cat(sprintf("%d%%", as.integer(rate * 100)))
        for (sp in seq_len(N_SPLITS)) {
          seed <- make_split_seed(sp, ds_name, rate, mech)
          res <- tryCatch(
            run_one_split(df, rate, mech, seed),
            error = function(e) {
              message(sprintf("\n  [ERR %s r=%.2f sp=%d] %s",
                              mech, rate, sp, e$message))
              NULL
            })
          if (is.null(res)) {
            cat("x")
            append_split_timing(ds_name, mech, rate, sp, NULL, status = "error")
            next
          }
          tm <- attr(res, "timing")
          res$Dataset  <- ds_name
          res$MissMech <- mech
          res$MissRate <- rate
          res$Split    <- sp
          ds_rows[[length(ds_rows) + 1L]] <- res
          append_split_timing(ds_name, mech, rate, sp, res, status = "ok")
          n_ok_sp <- n_ok_sp + 1L
          cat(sprintf(".%.0fs", if (!is.null(tm)) tm$sec_total else NA_real_))
        }
        cat(" ")
      }
      cat("done\n")
    }
  }
  
  ds_df <- dplyr::bind_rows(ds_rows)
  saveRDS(ds_df, ckpt_file)
  all_results[[ds_name]] <- ds_df
  elapsed <- proc.time()[3] - t0_ds
  timing_dataset_log[[length(timing_dataset_log) + 1L]] <- data.frame(
    Dataset = ds_name, mode = ds_mode, n = n, p = p, n_events = n_ev,
    status = "ok", n_splits_done = n_ok_sp,
    sec_total = as.numeric(elapsed),
    sec_per_split_mean = if (n_ok_sp > 0) as.numeric(elapsed) / n_ok_sp else NA_real_,
    stringsAsFactors = FALSE)
  write_csv(dplyr::bind_rows(timing_dataset_log),
            file.path(TIMING_DIR, "timing_by_dataset.csv"))
  cat(sprintf("  [Dataset total: %.1f sec / %.1f min | ~%.1f sec/split]  saved %s\n",
              elapsed, elapsed / 60,
              if (n_ok_sp > 0) elapsed / n_ok_sp else NA_real_,
              basename(ckpt_file)))
}

results <- dplyr::bind_rows(all_results)
if (nrow(results) == 0) stop("No results produced. Check data paths.")

# Sanitize CalSlope from older checkpoints / rare unstable fits (CCA+SurvKNN etc.)
if ("CalSlope" %in% names(results)) {
  n_bad <- sum(is.finite(results$CalSlope) & abs(results$CalSlope) > 20)
  if (n_bad > 0L) {
    message(sprintf(
      ">>> CalSlope: setting %d unstable |slope|>20 values to NA (ranking safety)",
      n_bad))
    results$CalSlope[is.finite(results$CalSlope) & abs(results$CalSlope) > 20] <- NA_real_
  }
}

# ---- 10b. Persist timing + cost summary ----
# Checkpoint-only resume: timing_split_log is often empty (no status column).
# Never assume status/sec_* exist; keep summarization defensive.
sec_run <- proc.time()[3] - t0_run
empty_split_timing <- data.frame(
  Dataset = character(0), MissMech = character(0), MissRate = numeric(0),
  Split = integer(0), status = character(0),
  sec_total = numeric(0), sec_mice = numeric(0),
  sec_within_clf = numeric(0), sec_mice_se = numeric(0),
  stringsAsFactors = FALSE
)
timing_split_df <- if (length(timing_split_log) > 0)
  dplyr::bind_rows(timing_split_log) else empty_split_timing
timing_ds_df <- if (length(timing_dataset_log) > 0)
  dplyr::bind_rows(timing_dataset_log) else
    data.frame(Dataset = character(0), status = character(0),
               sec_total = numeric(0), n_splits_done = integer(0),
               stringsAsFactors = FALSE)

write_csv(timing_split_df, file.path(TIMING_DIR, "timing_per_split.csv"))
write_csv(timing_split_df, file.path(OUTPUT_DIR, "timing_per_split.csv"))
write_csv(timing_ds_df, file.path(TIMING_DIR, "timing_by_dataset.csv"))
write_csv(timing_ds_df, file.path(OUTPUT_DIR, "timing_by_dataset.csv"))

ok_sp <- timing_split_df
if ("status" %in% names(ok_sp)) {
  ok_sp <- dplyr::filter(ok_sp, .data$status == "ok")
}
if ("sec_total" %in% names(ok_sp)) {
  ok_sp <- dplyr::filter(ok_sp, is.finite(.data$sec_total))
} else {
  ok_sp <- ok_sp[0, , drop = FALSE]
}
col_mean_safe <- function(df, col) {
  if (nrow(df) == 0L || !(col %in% names(df))) return(NA_real_)
  mean(df[[col]], na.rm = TRUE)
}
timing_summary <- data.frame(
  run_id = RUN_ID,
  finished_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
  sec_run_total = as.numeric(sec_run),
  min_run_total = as.numeric(sec_run) / 60,
  n_splits_logged = nrow(timing_split_df),
  n_splits_ok = nrow(ok_sp),
  sec_per_split_mean = if (nrow(ok_sp) > 0) mean(ok_sp$sec_total) else NA_real_,
  sec_per_split_median = if (nrow(ok_sp) > 0) stats::median(ok_sp$sec_total) else NA_real_,
  sec_per_split_min = if (nrow(ok_sp) > 0) min(ok_sp$sec_total) else NA_real_,
  sec_per_split_max = if (nrow(ok_sp) > 0) max(ok_sp$sec_total) else NA_real_,
  sec_mice_mean = col_mean_safe(ok_sp, "sec_mice"),
  sec_within_clf_mean = col_mean_safe(ok_sp, "sec_within_clf"),
  sec_mice_se_mean = col_mean_safe(ok_sp, "sec_mice_se"),
  approx_fits_per_split = COMP_STRUCT$approx_fit_counts_per_artificial_split$total_approx,
  mice_se_l0_cols = COMP_STRUCT$design$mice_se_l0_columns,
  stringsAsFactors = FALSE
)
write_csv(timing_summary, file.path(TIMING_DIR, "timing_summary.csv"))
write_csv(timing_summary, file.path(OUTPUT_DIR, "timing_summary.csv"))

# Human-readable cost report
cost_lines <- c(
  paste0("RUN_ID: ", RUN_ID),
  paste0("Finished: ", timing_summary$finished_at),
  sprintf("Total wall time: %.1f sec (%.2f min)", timing_summary$sec_run_total,
          timing_summary$min_run_total),
  sprintf("Splits OK/logged: %d / %d", timing_summary$n_splits_ok,
          timing_summary$n_splits_logged),
  sprintf("Per-split sec: mean=%.1f  median=%.1f  min=%.1f  max=%.1f",
          timing_summary$sec_per_split_mean, timing_summary$sec_per_split_median,
          timing_summary$sec_per_split_min, timing_summary$sec_per_split_max),
  sprintf("Per-split block means (sec): mice=%.1f | within-clf=%.1f | MICE_SE=%.1f",
          timing_summary$sec_mice_mean, timing_summary$sec_within_clf_mean,
          timing_summary$sec_mice_se_mean),
  sprintf("Approx survival fits / split: %s",
          timing_summary$approx_fits_per_split),
  sprintf("MICE_SE L0 columns: %s", timing_summary$mice_se_l0_cols),
  "",
  "Files:",
  "  computation_structure.txt / .rds",
  "  timing_per_split.csv",
  "  timing_by_dataset.csv",
  "  timing_summary.csv",
  "  timing/  (same copies)"
)
writeLines(cost_lines, file.path(TIMING_DIR, "computation_cost_report.txt"))
writeLines(cost_lines, file.path(OUTPUT_DIR, "computation_cost_report.txt"))
cat("\n>>> Timing / cost saved:\n")
cat(paste0("  ", cost_lines[1:8]), sep = "\n")
cat("\n")

# ---- 11. Summaries (always split by Classifier) ----
# Fallback audit columns (older checkpoints may lack them)
if (!"Fallback" %in% names(results)) {
  results$Fallback <- 0L
  results$FallbackType <- NA_character_
} else {
  results$Fallback[is.na(results$Fallback)] <- 0L
}
write_csv(results, file.path(OUTPUT_DIR, "impstack_raw_results.csv"))

fb_rows <- results %>% dplyr::filter(as.integer(Fallback) == 1L)
fb_summary <- results %>%
  dplyr::group_by(Method, FallbackType) %>%
  dplyr::summarise(
    n_rows = dplyr::n(),
    n_fallback = sum(as.integer(Fallback) == 1L, na.rm = TRUE),
    fallback_rate = mean(as.integer(Fallback) == 1L, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::arrange(dplyr::desc(n_fallback), Method)
fb_by_type <- results %>%
  dplyr::filter(as.integer(Fallback) == 1L) %>%
  dplyr::count(FallbackType, name = "n") %>%
  dplyr::arrange(dplyr::desc(n))
write_csv(fb_summary, file.path(OUTPUT_DIR, "fallback_summary_by_method.csv"))
write_csv(fb_by_type, file.path(OUTPUT_DIR, "fallback_counts_by_type.csv"))
if (nrow(fb_rows) > 0) {
  write_csv(fb_rows, file.path(OUTPUT_DIR, "fallback_events_detail.csv"))
}
cat(sprintf(
  "\n>>> Fallback audit: %d / %d result rows (%.2f%%)\n",
  nrow(fb_rows), nrow(results),
  100 * nrow(fb_rows) / max(1L, nrow(results))
))
if (nrow(fb_by_type) > 0) {
  cat("  By type:\n")
  print(as.data.frame(fb_by_type), row.names = FALSE)
} else {
  cat("  No fallbacks recorded in this run.\n")
}
cat("  saved: fallback_summary_by_method.csv, fallback_counts_by_type.csv\n")

# ImpStack_MICE meta coefficients / importances (per split) + means
sel_rows <- results %>%
  filter(Method == "ImpStack_MICE", !is.na(MetaCoefs))
if (nrow(sel_rows) > 0) {
  write_csv(
    sel_rows %>%
      select(Dataset, Classifier, MissMech, MissRate, Split,
             SelectedLearner, MetaCoefs, Cindex),
    file.path(OUTPUT_DIR, "impstack_meta_coefs_by_split.csv")
  )
  parse_w <- function(s) {
    parts <- strsplit(s, ";", fixed = TRUE)[[1]]
    kv <- strsplit(parts, "=", fixed = TRUE)
    data.frame(
      Engine = vapply(kv, `[`, character(1), 1),
      Weight = as.numeric(vapply(kv, `[`, character(1), 2)),
      stringsAsFactors = FALSE
    )
  }
  w_long <- sel_rows %>%
    select(Dataset, Classifier, MissMech, MissRate, Split, MetaCoefs) %>%
    mutate(pw = lapply(MetaCoefs, parse_w)) %>%
    tidyr::unnest(pw)
  w_mean <- w_long %>%
    group_by(Classifier, MissMech, MissRate, Engine) %>%
    summarise(mean_coef = mean(Weight, na.rm = TRUE),
              sd_coef = sd(Weight, na.rm = TRUE),
              .groups = "drop") %>%
    arrange(Classifier, MissMech, MissRate, Engine)
  write_csv(w_mean, file.path(OUTPUT_DIR, "impstack_meta_coefs_mean.csv"))
  cat("\n>>> ImpStack_MICE mean meta coefs/importances (overall):\n")
  print(
    w_long %>%
      group_by(Classifier, Engine) %>%
      summarise(mean_coef = round(mean(Weight, na.rm = TRUE), 4),
                .groups = "drop") %>%
      arrange(Classifier, desc(abs(mean_coef)))
  )
}

# Per-dataset means (over splits) — building block for grand mean + CI
summary_tbl <- results %>%
  group_by(Dataset, Classifier, MissMech, MissRate, Method) %>%
  summarise(
    Cindex_mean   = mean(Cindex,   na.rm = TRUE),
    IBS_mean      = mean(IBS,      na.rm = TRUE),
    CalSlope_mean = mean(CalSlope, na.rm = TRUE),
    .groups = "drop"
  )
write_csv(summary_tbl, file.path(OUTPUT_DIR, "impstack_summary_by_dataset.csv"))

ds_means <- summary_tbl %>%
  transmute(Dataset, Classifier, MissMech, MissRate, Method,
            Cindex = Cindex_mean,
            IBS = IBS_mean, CalSlope = CalSlope_mean)
write_csv(ds_means, file.path(OUTPUT_DIR, "impstack_dataset_means.csv"))

# Grand mean = mean of dataset means; 95% t-CI across datasets (n_datasets)
grand <- ds_means %>%
  group_by(Classifier, MissMech, MissRate, Method) %>%
  summarise(
    n_datasets = dplyr::n(),
    # CI from dataset means first (before overwriting with grand mean)
    Cindex_lo   = mean_ci_lo(Cindex),
    Cindex_hi   = mean_ci_hi(Cindex),
    Cindex      = mean(Cindex, na.rm = TRUE),
    IBS_lo      = mean_ci_lo(IBS),
    IBS_hi      = mean_ci_hi(IBS),
    IBS         = mean(IBS, na.rm = TRUE),
    CalSlope_lo = mean_ci_lo(CalSlope),
    CalSlope_hi = mean_ci_hi(CalSlope),
    CalSlope    = mean(CalSlope, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    Classifier = factor(Classifier, levels = SURV_NAMES),
    Method     = factor(Method, levels = METHOD_ORDER)
  ) %>%
  arrange(Classifier, MissMech, MissRate, Method)

write_csv(grand, file.path(OUTPUT_DIR, "impstack_grand_means.csv"))

# Separate CSV per classifier for easy reading
for (clf in SURV_NAMES) {
  g_clf <- grand %>% filter(Classifier == clf)
  write_csv(g_clf, file.path(OUTPUT_DIR, paste0("grand_", clf, ".csv")))
}

# ---- Best-only summaries (retain winner per metric → less table noise) ----
#   Cindex — higher better
#   IBS — lower better
#   CalSlope — closer to 1 better
pick_best_rows <- function(df, metric, higher_better, group_vars) {
  df %>%
    filter(is.finite(.data[[metric]])) %>%
    group_by(across(all_of(group_vars))) %>%
    group_modify(function(g, ...) {
      if (nrow(g) == 0) return(g[0, , drop = FALSE])
      best_val <- if (higher_better) max(g[[metric]], na.rm = TRUE)
      else min(g[[metric]], na.rm = TRUE)
      winners <- g[abs(g[[metric]] - best_val) < 1e-12, , drop = FALSE]
      meth <- as.character(winners$Method)
      if ("ImpStack_MICE" %in% meth) {
        winners <- winners[meth == "ImpStack_MICE", , drop = FALSE]
      } else {
        winners <- winners[1, , drop = FALSE]
      }
      winners
    }) %>%
    ungroup()
}

pick_best_cal_slope <- function(df, group_vars) {
  df %>%
    filter(is.finite(CalSlope)) %>%
    group_by(across(all_of(group_vars))) %>%
    group_modify(function(g, ...) {
      if (nrow(g) == 0) return(g[0, , drop = FALSE])
      i <- which.min(abs(g$CalSlope - 1))
      winners <- g[i, , drop = FALSE]
      meth <- as.character(winners$Method)
      # Prefer ImpStack on exact ties of |slope-1|
      d <- abs(g$CalSlope - 1)
      tied <- g[abs(d - d[i]) < 1e-12, , drop = FALSE]
      if ("ImpStack_MICE" %in% as.character(tied$Method)) {
        winners <- tied[as.character(tied$Method) == "ImpStack_MICE", ,
                        drop = FALSE][1, , drop = FALSE]
      }
      winners
    }) %>%
    ungroup()
}

gvars <- c("Classifier", "MissMech", "MissRate")
best_by_metric <- dplyr::bind_rows(
  pick_best_rows(grand, "Cindex", TRUE,  gvars) %>%
    transmute(Classifier, MissMech, MissRate, Metric = "Cindex",
              BestMethod = as.character(Method), BestValue = Cindex,
              Direction = "higher_better"),
  pick_best_rows(grand, "IBS",    FALSE, gvars) %>%
    transmute(Classifier, MissMech, MissRate, Metric = "IBS",
              BestMethod = as.character(Method), BestValue = IBS,
              Direction = "lower_better"),
  pick_best_cal_slope(grand, gvars) %>%
    transmute(Classifier, MissMech, MissRate, Metric = "CalSlope",
              BestMethod = as.character(Method), BestValue = CalSlope,
              Direction = "closer_to_1_better")
) %>% arrange(Classifier, MissMech, MissRate, Metric)
write_csv(best_by_metric, file.path(OUTPUT_DIR, "impstack_best_by_metric.csv"))

best_wide <- best_by_metric %>%
  filter(Metric == "Cindex") %>%
  select(Classifier, MissMech, MissRate,
         BestMethod_Cindex = BestMethod, BestValue_Cindex = BestValue) %>%
  left_join(
    best_by_metric %>% filter(Metric == "IBS") %>%
      select(Classifier, MissMech, MissRate,
             BestMethod_IBS = BestMethod, BestValue_IBS = BestValue),
    by = gvars) %>%
  left_join(
    best_by_metric %>% filter(Metric == "CalSlope") %>%
      select(Classifier, MissMech, MissRate,
             BestMethod_CalSlope = BestMethod, BestValue_CalSlope = BestValue),
    by = gvars)
write_csv(best_wide, file.path(OUTPUT_DIR, "impstack_best_retained_wide.csv"))

# Dataset-level best (mean over splits, then retain best method)
best_ds <- ds_means
dvars <- c("Dataset", "Classifier", "MissMech", "MissRate")
best_ds_long <- dplyr::bind_rows(
  pick_best_rows(best_ds, "Cindex", TRUE,  dvars) %>%
    transmute(Dataset, Classifier, MissMech, MissRate, Metric = "Cindex",
              BestMethod = as.character(Method), BestValue = Cindex),
  pick_best_rows(best_ds, "IBS",    FALSE, dvars) %>%
    transmute(Dataset, Classifier, MissMech, MissRate, Metric = "IBS",
              BestMethod = as.character(Method), BestValue = IBS),
  pick_best_cal_slope(best_ds, dvars) %>%
    transmute(Dataset, Classifier, MissMech, MissRate, Metric = "CalSlope",
              BestMethod = as.character(Method), BestValue = CalSlope)
) %>% arrange(Dataset, Classifier, MissMech, MissRate, Metric)
write_csv(best_ds_long, file.path(OUTPUT_DIR, "impstack_best_by_metric_dataset.csv"))

cat("\n>>> Best methods retained (grand; C↑ / IBS↓ / CalSlope~1):\n")
print(as.data.frame(best_wide), row.names = FALSE)
cat("  saved: impstack_best_by_metric.csv\n")
cat("  saved: impstack_best_retained_wide.csv\n")
cat("  saved: impstack_best_by_metric_dataset.csv\n")
cat("  note: metrics = Harrell C + IBS + CalSlope; grand CI = 95% t across datasets.\n")

cat("\n========== Grand means + 95% CI (by classifier) ==========\n")
print(as.data.frame(grand), row.names = FALSE)

# ---- 11a. Multi-dataset evaluation (primary summaries) ----
# Absolute grand means mix heterogeneous baselines (secondary / supplement).
# Primary quantities are WITHIN-DATASET comparable:
#   (i)  ranks → mean rank + Kendall W  (built in section 11b)
#   (ii) paired improvement vs REF_BASELINE on each dataset, then average
#   (iii) win rates (# datasets where method beats baseline / is best)
EVAL_DIR <- file.path(OUTPUT_DIR, "manuscript_eval")
dir.create(EVAL_DIR, recursive = TRUE, showWarnings = FALSE)

# Signed improvement so that POSITIVE = better than baseline on that dataset.
signed_improvement <- function(metric, method_val, ref_val) {
  if (identical(metric, "Cindex")) {
    return(method_val - ref_val)
  }
  if (identical(metric, "IBS")) {
    return(ref_val - method_val)  # lower IBS better
  }
  if (identical(metric, "CalSlope")) {
    return(abs(ref_val - 1) - abs(method_val - 1))  # closer to 1 better
  }
  rep(NA_real_, length(method_val))
}

metric_direction_note <- function(metric) {
  switch(metric,
         Cindex = "delta = method - baseline (higher C better)",
         IBS = "delta = baseline - method (lower IBS better)",
         CalSlope = "delta = |baseline-1| - |method-1| (closer to 1 better)",
         "delta undefined")
}

build_delta_long <- function(ds_df, metrics, baseline = REF_BASELINE) {
  keys <- c("Dataset", "Classifier", "MissMech", "MissRate")
  ref <- ds_df %>%
    filter(as.character(Method) == baseline) %>%
    select(all_of(keys), all_of(metrics))
  names(ref)[names(ref) %in% metrics] <- paste0("ref_", metrics)
  out_list <- list()
  for (met in metrics) {
    ref_col <- paste0("ref_", met)
    tmp <- ds_df %>%
      select(all_of(keys), Method, val = all_of(met)) %>%
      left_join(ref %>% select(all_of(keys), ref = all_of(ref_col)),
                by = keys) %>%
      mutate(
        Metric = met,
        Baseline = baseline,
        Improvement = signed_improvement(met, val, ref),
        BeatsBaseline = is.finite(Improvement) & Improvement > 0,
        Display = unname(METHOD_DISPLAY[as.character(Method)])
      ) %>%
      select(Dataset, Classifier, MissMech, MissRate, Method, Display,
             Metric, Baseline, MethodValue = val, BaselineValue = ref,
             Improvement, BeatsBaseline)
    out_list[[met]] <- tmp
  }
  dplyr::bind_rows(out_list)
}

ALL_EVAL_METRICS <- PRIMARY_METRICS
delta_by_dataset <- build_delta_long(ds_means, ALL_EVAL_METRICS, REF_BASELINE)
write_csv(delta_by_dataset,
          file.path(EVAL_DIR, "delta_vs_baseline_by_dataset.csv"))
write_csv(delta_by_dataset,
          file.path(OUTPUT_DIR, "delta_vs_baseline_by_dataset.csv"))

# Mean paired improvement across datasets (+ 95% t-CI)
delta_summary <- delta_by_dataset %>%
  filter(as.character(Method) != Baseline) %>%
  group_by(Classifier, MissMech, MissRate, Method, Display, Metric, Baseline) %>%
  summarise(
    n_datasets = sum(is.finite(Improvement)),
    mean_improvement = mean(Improvement, na.rm = TRUE),
    lo = mean_ci_lo(Improvement),
    hi = mean_ci_hi(Improvement),
    win_rate_vs_baseline = mean(BeatsBaseline, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(Method = factor(as.character(Method), levels = METHOD_ORDER)) %>%
  arrange(Metric, Classifier, MissMech, MissRate, Method)
write_csv(delta_summary, file.path(EVAL_DIR, "delta_vs_baseline_summary.csv"))
write_csv(delta_summary, file.path(OUTPUT_DIR, "delta_vs_baseline_summary.csv"))

# Best-method win rate across datasets (primary decision quantity)
win_best_rows <- list()
for (met in ALL_EVAL_METRICS) {
  higher_better <- identical(met, "Cindex")
  cal_mode <- identical(met, "CalSlope")
  for (clf in unique(as.character(ds_means$Classifier))) {
    for (mech in unique(as.character(ds_means$MissMech))) {
      for (rate in unique(ds_means$MissRate[as.character(ds_means$MissMech) == mech])) {
        sub <- ds_means %>%
          filter(Classifier == clf, MissMech == mech,
                 abs(MissRate - rate) < 1e-6) %>%
          select(Dataset, Method, val = all_of(met))
        if (nrow(sub) == 0) next
        score <- if (cal_mode) -abs(sub$val - 1) else if (higher_better) sub$val else -sub$val
        sub$score <- score
        best <- sub %>%
          filter(is.finite(score)) %>%
          group_by(Dataset) %>%
          summarise(
            BestMethod = {
              mx <- max(score, na.rm = TRUE)
              w <- as.character(Method)[abs(score - mx) < 1e-12]
              if ("ImpStack_MICE" %in% w) "ImpStack_MICE" else w[[1]]
            },
            .groups = "drop"
          )
        tab <- as.data.frame(table(BestMethod = best$BestMethod),
                             stringsAsFactors = FALSE)
        names(tab)[2] <- "n_wins"
        tab$n_datasets <- nrow(best)
        tab$win_rate_best <- tab$n_wins / tab$n_datasets
        tab$Classifier <- clf
        tab$MissMech <- mech
        tab$MissRate <- rate
        tab$Metric <- met
        win_best_rows[[length(win_best_rows) + 1L]] <- tab
      }
    }
  }
}
win_best <- dplyr::bind_rows(win_best_rows) %>%
  mutate(
    Display = unname(METHOD_DISPLAY[as.character(BestMethod)]),
    Method = factor(as.character(BestMethod), levels = METHOD_ORDER)
  ) %>%
  select(Classifier, MissMech, MissRate, Metric, Method, Display,
         n_wins, n_datasets, win_rate_best) %>%
  arrange(Metric, Classifier, MissMech, MissRate, desc(win_rate_best))
write_csv(win_best, file.path(EVAL_DIR, "winrate_best_by_dataset.csv"))
write_csv(win_best, file.path(OUTPUT_DIR, "winrate_best_by_dataset.csv"))

# Compact primary tables (Harrell C / IBS / CalSlope only)
primary_delta <- delta_summary %>%
  filter(Metric %in% PRIMARY_METRICS)
primary_wins <- win_best %>%
  filter(Metric %in% PRIMARY_METRICS)
write_csv(primary_delta, file.path(EVAL_DIR, "PRIMARY_delta_vs_baseline.csv"))
write_csv(primary_wins, file.path(EVAL_DIR, "PRIMARY_winrate_best.csv"))

# ImpStack focus card (easy for Results writing)
impstack_card <- delta_summary %>%
  filter(as.character(Method) == "ImpStack_MICE",
         Metric %in% PRIMARY_METRICS) %>%
  transmute(
    Classifier, MissMech, MissRate, Metric,
    mean_delta_vs_MeanSI = mean_improvement,
    delta_lo = lo, delta_hi = hi,
    win_rate_vs_MeanSI = win_rate_vs_baseline,
    n_datasets
  ) %>%
  arrange(Metric, Classifier, MissMech, MissRate)
write_csv(impstack_card, file.path(EVAL_DIR, "PRIMARY_ImpStack_vs_MeanSI_card.csv"))

eval_readme <- c(
  "# Evaluation outputs",
  "",
  "Design: datasets are heterogeneous, so absolute grand-mean C-index is SECONDARY.",
  "PRIMARY quantities (use in main text):",
  "  1. Within-dataset ranks → mean rank + Kendall W  (see latex/ + rank_mean_all_methods.csv)",
  paste0("  2. Paired improvement vs ", REF_BASELINE, " on each dataset, then mean across datasets"),
  "  3. Win rate: fraction of datasets where a method is best / beats baseline",
  "",
  paste0("Primary metrics: ", paste(PRIMARY_METRICS, collapse = ", ")),
  "",
  "Key files:",
  "  PRIMARY_ImpStack_vs_MeanSI_card.csv  — Results-section helper for ImpStack",
  "  PRIMARY_delta_vs_baseline.csv        — all methods, primary metrics",
  "  PRIMARY_winrate_best.csv             — who wins how often",
  "  delta_vs_baseline_by_dataset.csv     — full paired deltas",
  "",
  "Grand means remain in ../impstack_grand_means.csv (descriptive / supplement).",
  "See EVALUATION.md in the project root."
)
writeLines(eval_readme, file.path(EVAL_DIR, "README_eval.txt"))

cat("\n>>> Meaningful evaluation (PRIMARY) written to:", EVAL_DIR, "\n")
cat("  Reference baseline:", REF_BASELINE, "\n")
cat("  Primary metrics   :", paste(PRIMARY_METRICS, collapse = ", "), "\n")
if (nrow(impstack_card) > 0) {
  cat("\n>>> ImpStack mean delta vs MeanSI (Harrell C; sample):\n")
  print(as.data.frame(
    impstack_card %>% filter(Metric == "Cindex") %>%
      mutate(MissRate = MissRate * 100) %>%
      select(Classifier, MissMech, MissRate, mean_delta_vs_MeanSI,
             win_rate_vs_MeanSI) %>%
      head(12)
  ), row.names = FALSE)
}


# ---- 12. Visualizations (robust saves: one failure must not kill the rest) ----
if (!exists("FIG_DIR") || !dir.exists(FIG_DIR)) {
  FIG_DIR <- file.path(OUTPUT_DIR, "figures")
  dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)
}
cat("\n>>> Building figures in:", FIG_DIR, "\n")

# Safe saver: try PNG first (most reliable), then PDF. Never stops the pipeline.
save_plot_safe <- function(plot, stem, width = 10, height = 7, dpi = 200,
                           also = NULL) {
  dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)
  paths <- character(0)
  png_path <- file.path(FIG_DIR, paste0(stem, ".png"))
  pdf_path <- file.path(FIG_DIR, paste0(stem, ".pdf"))
  
  png_ok <- FALSE
  for (dev in list(
    function(f) ggsave(f, plot, width = width, height = height, dpi = dpi,
                       device = "png"),
    function(f) {
      grDevices::png(f, width = width, height = height, units = "in", res = dpi)
      print(plot); grDevices::dev.off()
    }
  )) {
    ok <- tryCatch({ dev(png_path); TRUE }, error = function(e) {
      message("  [PNG fail] ", stem, ": ", conditionMessage(e)); FALSE
    })
    if (isTRUE(ok) && file.exists(png_path) && file.info(png_path)$size > 0) {
      png_ok <- TRUE; break
    }
  }
  
  pdf_ok <- tryCatch({
    ggsave(pdf_path, plot, width = width, height = height, device = grDevices::pdf)
    file.exists(pdf_path) && file.info(pdf_path)$size > 0
  }, error = function(e) {
    message("  [PDF fail] ", stem, ": ", conditionMessage(e))
    FALSE
  })
  
  if (png_ok) {
    paths <- c(paths, png_path)
    cat("  saved:", basename(png_path), "\n")
  }
  if (isTRUE(pdf_ok)) {
    paths <- c(paths, pdf_path)
    cat("  saved:", basename(pdf_path), "\n")
  }
  if (!png_ok && !isTRUE(pdf_ok)) {
    message("  [WARN] could not save either PNG or PDF for: ", stem)
  }
  
  # Optional extra copies (e.g. legacy names in OUTPUT_DIR)
  if (!is.null(also) && length(also) > 0) {
    for (extra in also) {
      tryCatch({
        if (grepl("\\.png$", extra, ignore.case = TRUE) && png_ok) {
          file.copy(png_path, extra, overwrite = TRUE)
        } else if (grepl("\\.pdf$", extra, ignore.case = TRUE) && isTRUE(pdf_ok)) {
          file.copy(pdf_path, extra, overwrite = TRUE)
        } else if (png_ok) {
          ggsave(extra, plot, width = width, height = height, dpi = dpi, device = "png")
        }
      }, error = function(e) message("  [extra fail] ", extra, ": ",
                                     conditionMessage(e)))
    }
  }
  invisible(paths)
}

# Pretty labels (legend / axes). Keys = literal Method codes.
METHOD_LABELS <- METHOD_DISPLAY

theme_paper <- function(base_size = 11) {
  # Never call bare margin() here: randomForest masks ggplot2::margin and errors with
  #   argument "observed" is missing, with no default
  theme_bw(base_size = base_size) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(colour = "grey88", linewidth = 0.3),
      strip.background = element_rect(fill = "white", colour = "grey40"),
      strip.text = element_text(face = "bold", size = rel(0.9)),
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.text = element_text(size = rel(0.85)),
      plot.title = element_text(face = "bold", size = rel(1.05), hjust = 0),
      plot.subtitle = element_text(colour = "grey35", size = rel(0.9), hjust = 0,
                                   margin = ggplot2::margin(b = 8)),
      plot.caption = element_text(colour = "grey45", size = rel(0.8), hjust = 0),
      axis.title = element_text(size = rel(0.95)),
      axis.text = element_text(colour = "grey20")
    )
}

# Colour/linetype scales keyed by display label (for legend)
colour_by_label <- setNames(unname(METHOD_COLORS[METHOD_ORDER]),
                            unname(METHOD_DISPLAY[METHOD_ORDER]))
ltype_by_label <- setNames(
  ifelse(METHOD_ORDER == "ImpStack_MICE", "solid",
         ifelse(METHOD_ORDER == "MICE_SE", "twodash", "solid")),
  unname(METHOD_DISPLAY[METHOD_ORDER])
)

fig_ok <- tryCatch({
  if (!exists("grand") || !is.data.frame(grand) || nrow(grand) == 0) {
    stop("No 'grand' results table available — experiment must finish before figures.")
  }
  
  method_cols <- METHOD_COLORS  # keyed by Method code
  
  add_method_label <- function(df) {
    df %>%
      mutate(
        Method = factor(as.character(Method), levels = METHOD_ORDER),
        MethodLabel = factor(
          dplyr::recode(as.character(Method), !!!as.list(METHOD_DISPLAY)),
          levels = unname(METHOD_DISPLAY[METHOD_ORDER])
        ),
        is_stack = as.character(Method) == "ImpStack_MICE"
      )
  }
  
  plot_df <- grand %>%
    pivot_longer(cols = c(Cindex, IBS, CalSlope),
                 names_to = "Metric", values_to = "Value") %>%
    add_method_label() %>%
    mutate(
      Metric = factor(Metric,
                      levels = c("Cindex", "IBS", "CalSlope"),
                      labels = c("Harrell C-index", "IBS",
                                 "Cal. slope")),
      Method_short = MethodLabel,
      Rate_lab = ifelse(as.character(MissMech) == "NATURAL",
                        "NATURAL",
                        paste0(as.integer(round(MissRate * 100)), "%"))
    )
  
  # ---- A. Trajectory overview (Stacking bold overlay) ----
  # All methods in colour legend; Stacking redrawn thicker in blue.
  p_traj_all <- ggplot(plot_df %>% filter(!is.na(MethodLabel)),
                       aes(x = MissRate, y = Value,
                           colour = MethodLabel, linetype = MethodLabel,
                           group = MethodLabel)) +
    geom_line(linewidth = 0.55, alpha = 0.9) +
    geom_point(size = 1.1, alpha = 0.9) +
    geom_line(data = plot_df %>% filter(is_stack),
              aes(x = MissRate, y = Value, group = MethodLabel),
              colour = METHOD_COLORS[["ImpStack_MICE"]],
              linewidth = 1.7, linetype = "solid", inherit.aes = FALSE) +
    facet_grid(Metric ~ Classifier + MissMech, scales = "free_y") +
    scale_colour_manual(values = colour_by_label, drop = FALSE) +
    scale_linetype_manual(values = ltype_by_label, drop = FALSE) +
    scale_x_continuous(labels = scales::percent_format(accuracy = 1),
                       breaks = MISS_RATES) +
    labs(title = "Imputation stacking benchmark - trajectories",
         subtitle = "Stacking (blue, bold) = ImpStack_MICE · MIE-SE = cross-classifier competitor",
         caption = "C↑ better · IBS↓ better · CalSlope≈1 better · Grand mean over datasets",
         x = "Missingness rate", y = NULL) +
    theme_paper(9) +
    theme(legend.text = element_text(size = 7),
          strip.text = element_text(size = 6.5),
          axis.text.x = element_text(angle = 45, hjust = 1, size = 6)) +
    guides(colour = guide_legend(nrow = 3, override.aes = list(linewidth = 1.2)),
           linetype = guide_legend(nrow = 3))
  
  save_plot_safe(p_traj_all, "01_trajectories_all", width = 16, height = 14,
                 also = file.path(OUTPUT_DIR, "impstack_all_classifiers.pdf"))
  
  # ---- B. Per-classifier trajectories (Stacking bold) ----
  for (clf in SURV_NAMES) {
    pdf_clf <- plot_df %>% filter(Classifier == clf)
    if (nrow(pdf_clf) == 0) next
    p_clf <- ggplot(pdf_clf %>% filter(!is.na(MethodLabel)),
                    aes(x = MissRate, y = Value,
                        colour = MethodLabel, linetype = MethodLabel,
                        group = MethodLabel)) +
      geom_line(linewidth = 0.75, alpha = 0.9) +
      geom_point(size = 1.7) +
      geom_line(data = pdf_clf %>% filter(is_stack),
                aes(x = MissRate, y = Value, group = MethodLabel),
                colour = METHOD_COLORS[["ImpStack_MICE"]],
                linewidth = 1.9, linetype = "solid", inherit.aes = FALSE) +
      facet_grid(Metric ~ MissMech, scales = "free_y") +
      scale_colour_manual(values = colour_by_label, drop = FALSE) +
      scale_linetype_manual(values = ltype_by_label, drop = FALSE) +
      scale_x_continuous(labels = scales::percent_format(accuracy = 1),
                         breaks = MISS_RATES) +
      labs(title = paste0(clf, " - within-classifier comparison"),
           subtitle = "Stacking (blue, bold) highlighted · fixed method colours",
           caption = "C↑ · IBS↓ · CalSlope≈1 · see CSV for 95% CIs across datasets",
           x = "Missingness rate", y = NULL) +
      theme_paper(11) +
      guides(colour = guide_legend(nrow = 3, override.aes = list(linewidth = 1.3)),
             linetype = guide_legend(nrow = 3))
    save_plot_safe(p_clf, paste0("02_traj_", clf), width = 12, height = 11,
                   also = file.path(OUTPUT_DIR, paste0("impstack_", clf, ".pdf")))
  }
  
  # ---- B2. Per-dataset C-index only, one figure per classifier × mechanism ----
  if (exists("ds_means") && is.data.frame(ds_means) && nrow(ds_means) > 0) {
    for (clf in SURV_NAMES) {
      for (mech in unique(as.character(ds_means$MissMech))) {
        ds_plot <- ds_means %>%
          filter(as.character(Classifier) == clf,
                 as.character(MissMech) == mech,
                 is.finite(Cindex)) %>%
          add_method_label()
        if (nrow(ds_plot) == 0) next
        n_ds_fac <- length(unique(ds_plot$Dataset))
        ncol_f <- if (n_ds_fac <= 4) 2 else if (n_ds_fac <= 9) 3 else 4
        nrow_f <- ceiling(n_ds_fac / ncol_f)
        
        p_ds <- ggplot(ds_plot %>% filter(!is.na(MethodLabel)),
                       aes(x = MissRate * 100, y = Cindex,
                           colour = MethodLabel, linetype = MethodLabel,
                           group = MethodLabel)) +
          geom_line(linewidth = 0.55, alpha = 0.85) +
          geom_point(size = 1.5) +
          geom_line(data = ds_plot %>% filter(is_stack),
                    aes(x = MissRate * 100, y = Cindex, group = MethodLabel),
                    colour = METHOD_COLORS[["ImpStack_MICE"]],
                    linewidth = 1.9, linetype = "solid",
                    inherit.aes = FALSE) +
          facet_wrap(~Dataset, nrow = nrow_f, ncol = ncol_f, scales = "free_y") +
          scale_colour_manual(values = colour_by_label, drop = FALSE) +
          scale_linetype_manual(values = ltype_by_label, drop = FALSE) +
          scale_x_continuous(breaks = MISS_RATES * 100,
                             labels = paste0(as.integer(MISS_RATES * 100), "%")) +
          labs(title = sprintf("%s — dataset-level C-index under %s", clf, mech),
               subtitle = "Colour = imputation method; Stacking bold blue",
               caption = "Per-dataset mean over splits · Harrell C-index only",
               x = "Missingness Rate (%)", y = "Mean Harrell C-index") +
          theme_paper(8) +
          theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7)) +
          guides(colour = guide_legend(nrow = 3,
                                       override.aes = list(linewidth = 1.2)),
                 linetype = guide_legend(nrow = 3))
        stem <- sprintf("figB_per_dataset_cindex_%s_%s", clf, tolower(mech))
        save_plot_safe(p_ds, stem,
                       width = max(12, 4 * ncol_f),
                       height = max(8, 4 * nrow_f), dpi = 220)
      }
    }
  } else {
    message("  [skip figB] ds_means missing — per-dataset panels not built")
  }
  
  # ---- C/D. Heatmaps ----
  hm_c <- grand %>%
    mutate(
      Method_short = factor(METHOD_LABELS[as.character(Method)],
                            levels = rev(METHOD_LABELS[METHOD_ORDER])),
      Rate_lab = factor(paste0(as.integer(MissRate * 100), "%"),
                        levels = paste0(as.integer(MISS_RATES * 100), "%"))
    )
  
  p_hm_c <- ggplot(hm_c, aes(x = Rate_lab, y = Method_short, fill = Cindex)) +
    geom_tile(colour = "white", linewidth = 0.4) +
    geom_text(aes(label = sprintf("%.3f", Cindex)), size = 2.2, colour = "grey15") +
    facet_grid(MissMech ~ Classifier) +
    scale_fill_gradientn(
      colours = c("#F7FCF5", "#A1D99B", "#31A354", "#006D2C"),
      name = "Harrell C"
    ) +
    labs(title = "2D scan - Harrell C-index across methods x missingness",
         subtitle = "Each cell = grand-mean C-index; darker green = better discrimination",
         caption = "Source: grand means over datasets · Within-classifier blocks",
         x = "Missingness rate", y = NULL) +
    theme_paper(10) +
    theme(panel.grid = element_blank(),
          axis.text.y = element_text(size = 8),
          legend.position = "right")
  save_plot_safe(p_hm_c, "03_heatmap_Cindex_2Dscan", width = 14, height = 8, dpi = 220)
  
  p_hm_ibs <- ggplot(hm_c, aes(x = Rate_lab, y = Method_short, fill = IBS)) +
    geom_tile(colour = "white", linewidth = 0.4) +
    geom_text(aes(label = sprintf("%.3f", IBS)), size = 2.2, colour = "grey15") +
    facet_grid(MissMech ~ Classifier) +
    scale_fill_gradientn(
      colours = c("#08306B", "#2171B5", "#6BAED6", "#DEEBF7", "#FFF5F0"),
      name = "IBS"
    ) +
    labs(title = "2D scan - Integrated Brier Score across methods x missingness",
         subtitle = "Each cell = grand-mean IBS; cooler / darker blue = better (lower IBS)",
         caption = "Source: grand means · Cox native S(t) or Cox-calibration of risk score",
         x = "Missingness rate", y = NULL) +
    theme_paper(10) +
    theme(panel.grid = element_blank(),
          axis.text.y = element_text(size = 8),
          legend.position = "right")
  save_plot_safe(p_hm_ibs, "04_heatmap_IBS_2Dscan", width = 14, height = 8, dpi = 220)
  
  
  p_hm_cal <- ggplot(hm_c, aes(x = Rate_lab, y = Method_short, fill = CalSlope)) +
    geom_tile(colour = "white", linewidth = 0.3) +
    geom_text(aes(label = sprintf("%.2f", CalSlope)), size = 2.2, colour = "grey15") +
    facet_grid(MissMech ~ Classifier) +
    scale_fill_gradient2(
      low = "#D73027", mid = "#FFFFBF", high = "#1A9850",
      midpoint = 1, name = "Cal.\nslope"
    ) +
    labs(title = "2D scan - Calibration slope (methods x missingness)",
         subtitle = "Test Cox slope of final risk score; target = 1 (yellow/green near 1)",
         caption = "Same score as C-index · Not the train-only IBS calibration map",
         x = "Missingness rate", y = NULL) +
    theme_paper(10) +
    theme(panel.grid = element_blank(),
          axis.text.y = element_text(size = 8),
          legend.position = "right")
  save_plot_safe(p_hm_cal, "04b_heatmap_CalSlope_2Dscan", width = 14, height = 8, dpi = 220)
  
  # ---- E. Delta heatmaps ----
  delta_c <- grand %>%
    select(Classifier, MissMech, MissRate, Method, Cindex) %>%
    tidyr::pivot_wider(names_from = Method, values_from = Cindex)
  
  need_cols <- c("ImpStack_MICE", "MeanSI", "RFI")
  if (!all(need_cols %in% names(delta_c))) {
    message("  [skip 05] missing columns for delta plot: ",
            paste(setdiff(need_cols, names(delta_c)), collapse = ", "))
  } else {
    delta_c <- delta_c %>%
      mutate(
        StackMICE_vs_MeanSI = ImpStack_MICE - MeanSI,
        StackMICE_vs_RFI    = ImpStack_MICE - RFI,
        Rate_lab = factor(paste0(as.integer(MissRate * 100), "%"),
                          levels = paste0(as.integer(MISS_RATES * 100), "%"))
      )
    delta_long <- delta_c %>%
      select(Classifier, MissMech, Rate_lab,
             StackMICE_vs_MeanSI, StackMICE_vs_RFI) %>%
      pivot_longer(cols = starts_with("Stack"),
                   names_to = "Contrast", values_to = "Delta") %>%
      mutate(
        Contrast = factor(Contrast,
                          levels = c("StackMICE_vs_MeanSI", "StackMICE_vs_RFI"),
                          labels = c("ImpStack_MICE - MeanSI",
                                     "ImpStack_MICE - RFI")),
        lbl_col = ifelse(abs(Delta) > 0.015, "white", "grey10")
      )
    p_delta <- ggplot(delta_long,
                      aes(x = Rate_lab, y = Classifier, fill = Delta)) +
      geom_tile(colour = "white", linewidth = 0.5) +
      geom_text(aes(label = sprintf("%+.3f", Delta), colour = lbl_col), size = 2.8) +
      scale_colour_identity() +
      facet_grid(Contrast ~ MissMech) +
      scale_fill_gradient2(
        low = "#B2182B", mid = "#F7F7F7", high = "#2166AC",
        midpoint = 0, name = "d C-index"
      ) +
      labs(title = "2D scan - ImpStack_MICE advantage (d Harrell C-index)",
           subtitle = "Positive (blue) = stacking better; negative (red) = worse",
           caption = "ImpStack_MICE = per-copy CoxPH meta on PMM/RF/CART/NORM/MIDASTOUCH, then average m stacks",
           x = "Missingness rate", y = NULL) +
      theme_paper(11) +
      theme(panel.grid = element_blank(), legend.position = "right")
    save_plot_safe(p_delta, "05_heatmap_ImpStack_delta_2Dscan",
                   width = 11, height = 7, dpi = 220)
  }
  # ---- E2. PRIMARY: mean paired delta vs REF_BASELINE (dataset-level, then mean) ----
  if (exists("delta_summary") && is.data.frame(delta_summary) && nrow(delta_summary) > 0) {
    dplot <- delta_summary %>%
      filter(Metric == "Cindex",
             as.character(Method) %in% c("ImpStack_MICE", "PMM", "RFI", "MICE_SE")) %>%
      mutate(
        Rate_lab = factor(paste0(as.integer(MissRate * 100), "%"),
                          levels = paste0(as.integer(MISS_RATES * 100), "%")),
        MethodLabel = unname(METHOD_DISPLAY[as.character(Method)])
      )
    if (nrow(dplot) > 0) {
      p_md <- ggplot(dplot, aes(x = Rate_lab, y = mean_improvement,
                                colour = MethodLabel, group = MethodLabel)) +
        geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50") +
        geom_line(linewidth = 0.8) +
        geom_point(size = 2) +
        facet_grid(MissMech ~ Classifier) +
        scale_colour_manual(values = colour_by_label, name = NULL) +
        labs(title = "PRIMARY: mean paired gain in Harrell C vs MeanSI",
             subtitle = "Within each dataset: method - MeanSI; then average across datasets (comparable)",
             caption = "Positive = better discrimination than mean imputation on average",
             x = "Missingness rate", y = "Mean delta C-index") +
        theme_paper(11) +
        theme(legend.position = "bottom")
      save_plot_safe(p_md, "08_PRIMARY_mean_delta_Cindex_vs_MeanSI",
                     width = 12, height = 7, dpi = 220)
    }
  }
  
  if (exists("win_best") && is.data.frame(win_best) && nrow(win_best) > 0) {
    wplot <- win_best %>%
      filter(Metric == "Cindex",
             as.character(Method) == "ImpStack_MICE") %>%
      mutate(
        Rate_lab = factor(paste0(as.integer(MissRate * 100), "%"),
                          levels = paste0(as.integer(MISS_RATES * 100), "%"))
      )
    if (nrow(wplot) > 0) {
      p_wr <- ggplot(wplot, aes(x = Rate_lab, y = win_rate_best, fill = Classifier)) +
        geom_col(position = "dodge", width = 0.7) +
        facet_wrap(~ MissMech) +
        scale_y_continuous(limits = c(0, 1), labels = scales::percent_format(accuracy = 1)) +
        labs(title = "PRIMARY: ImpStack win rate (Harrell C)",
             subtitle = "Fraction of datasets where ImpStack_MICE has the best C-index",
             x = "Missingness rate", y = "Win rate across datasets") +
        theme_paper(11) +
        theme(legend.position = "bottom")
      save_plot_safe(p_wr, "09_PRIMARY_ImpStack_winrate_Cindex",
                     width = 10, height = 5, dpi = 220)
    }
  }
  
  
  # ---- F. Rank heatmap ----
  rank_df <- grand %>%
    group_by(Classifier, MissMech, MissRate) %>%
    mutate(Rank = rank(-Cindex, ties.method = "average")) %>%
    ungroup() %>%
    mutate(
      Method_short = factor(METHOD_LABELS[as.character(Method)],
                            levels = rev(METHOD_LABELS[METHOD_ORDER])),
      Rate_lab = factor(paste0(as.integer(MissRate * 100), "%"),
                        levels = paste0(as.integer(MISS_RATES * 100), "%"))
    )
  p_rank <- ggplot(rank_df, aes(x = Rate_lab, y = Method_short, fill = Rank)) +
    geom_tile(colour = "white", linewidth = 0.35) +
    geom_text(aes(label = sprintf("%.0f", Rank)), size = 2.4) +
    facet_grid(MissMech ~ Classifier) +
    scale_fill_gradientn(
      colours = c("#1A9850", "#FEE08B", "#D73027"),
      name = "Rank\n(1=best)",
      breaks = c(1, 4, 7, length(METHOD_ORDER)),
      limits = c(1, length(METHOD_ORDER))
    ) +
    labs(title = "Descriptive: ranks of grand-mean Harrell C",
         subtitle = "Secondary display (absolute grands mix datasets). PRIMARY ranks: latex/ + rank_mean_all_methods.csv",
         caption = "Prefer within-dataset mean ranks / Kendall W for primary claims",
         x = "Missingness rate", y = NULL) +
    theme_paper(10) +
    theme(panel.grid = element_blank(),
          axis.text.y = element_text(size = 8),
          legend.position = "right")
  save_plot_safe(p_rank, "06_heatmap_ranks_2Dscan", width = 14, height = 8, dpi = 220)
  
  # ---- G. Kendall W ----
  if (exists("kendall_df") && is.data.frame(kendall_df) && nrow(kendall_df) > 0) {
    kw_plot <- kendall_df %>%
      filter(Metric == "Cindex") %>%
      mutate(
        Rate_lab = factor(paste0(as.integer(MissRate * 100), "%"),
                          levels = paste0(as.integer(MISS_RATES * 100), "%")),
        W_lab = sprintf("%.2f%s", Kendall_W, Signif)
      )
    p_kw <- ggplot(kw_plot, aes(x = Rate_lab, y = Classifier, fill = Kendall_W)) +
      geom_tile(colour = "white", linewidth = 0.5) +
      geom_text(aes(label = W_lab), size = 3) +
      facet_wrap(~ MissMech) +
      scale_fill_gradientn(
        colours = c("#F7FCF5", "#74C476", "#238B45", "#00441B"),
        name = "Kendall W", limits = c(0, 1)
      ) +
      labs(title = "2D scan - Kendall's W (C-index rank concordance)",
           subtitle = "Within each classifier: agreement of method rankings across datasets",
           caption = "Objects = methods; judges = datasets",
           x = "Missingness rate", y = NULL) +
      theme_paper(11) +
      theme(panel.grid = element_blank(), legend.position = "right")
    save_plot_safe(p_kw, "07_heatmap_KendallW_2Dscan", width = 10, height = 5, dpi = 220)
  } else {
    message("  [skip 07] kendall_df missing or empty")
  }
  
  TRUE
}, error = function(e) {
  message(">>> Figure section error: ", conditionMessage(e))
  FALSE
})

saved_figs <- list.files(FIG_DIR, pattern = "\\.(png|pdf)$", full.names = FALSE)
cat("\nFigures directory:", FIG_DIR, "\n")
cat("Files saved (", length(saved_figs), "):\n", sep = "")
if (length(saved_figs) == 0) {
  cat("  (none — check errors above; also ensure the main experiment finished and wrote grand means)\n")
} else {
  for (f in sort(saved_figs)) cat("  -", f, "\n")
}


# ---- 11b. Kendall's W ranks + LaTeX export (errors must not block figures) ----
tryCatch({
  # Within each Classifier × MissMech × MissRate × Metric:
  #   objects = methods, judges = datasets.
  # Higher-better: Cindex → rank(-score)
  # Lower-better:  IBS    → rank( score)
  LATEX_DIR <- file.path(OUTPUT_DIR, "latex")
  dir.create(LATEX_DIR, recursive = TRUE, showWarnings = FALSE)
  
  latex_preamble <- function(pt = 10) {
    c(
      sprintf("\\documentclass[%dpt]{article}", pt),
      "\\usepackage[margin=1.5cm,a4paper]{geometry}",
      "\\usepackage{booktabs,multirow,array,graphicx}",
      "\\usepackage{amsmath}",
      "\\begin{document}",
      ""
    )
  }
  
  # Dataset-level means (average over splits) — input to ranking
  ds_level <- ds_means
  
  compute_kendall_w <- function(rank_mat) {
    # rank_mat: rows = methods, cols = datasets
    valid_cols <- colSums(!is.na(rank_mat)) >= 2
    valid_rows <- rowSums(!is.na(rank_mat[, valid_cols, drop = FALSE])) >= 2
    rm <- rank_mat[valid_rows, valid_cols, drop = FALSE]
    n_obj <- nrow(rm); m_judges <- ncol(rm)
    out <- list(W = NA_real_, chi2 = NA_real_, df = NA_real_,
                p = NA_real_, signif = "ns", n_obj = n_obj, m = m_judges)
    if (n_obj < 2 || m_judges < 2) return(out)
    rs <- rowSums(rm, na.rm = TRUE)
    mean_rs <- m_judges * (n_obj + 1) / 2
    S <- sum((rs - mean_rs)^2)
    W <- 12 * S / (m_judges^2 * (n_obj^3 - n_obj))
    chi2 <- m_judges * (n_obj - 1) * W
    df <- n_obj - 1
    p <- pchisq(chi2, df = df, lower.tail = FALSE)
    sig <- if (p < 0.001) "***" else if (p < 0.01) "**" else if (p < 0.05) "*" else "ns"
    list(W = W, chi2 = chi2, df = df, p = p, signif = sig,
         n_obj = n_obj, m = m_judges)
  }
  
  # Build rank matrix + Kendall W for one condition (within one classifier)
  make_rank_block <- function(clf, mech, rate, metric) {
    higher_better <- identical(metric, "Cindex")
    sub <- ds_level %>%
      filter(Classifier == clf, MissMech == mech,
             abs(MissRate - rate) < 1e-6) %>%
      select(Dataset, Method, val = all_of(metric))
    # CalSlope: rank by |slope - 1| (closer to 1 = better)
    if (identical(metric, "CalSlope")) {
      sub <- sub %>% mutate(val = abs(val - 1))
      higher_better <- FALSE
    }
    if (nrow(sub) < 2) return(NULL)
    
    # Force every METHOD_ORDER method into the rank table (incl. MICE_SE / MIE-SE),
    # even if a condition has missing scores for some datasets.
    wide <- tidyr::pivot_wider(sub, names_from = Dataset, values_from = val)
    datasets <- setdiff(names(wide), "Method")
    if (length(datasets) < 2) return(NULL)
    for (m in METHOD_ORDER) {
      if (!(m %in% as.character(wide$Method))) {
        wide <- dplyr::bind_rows(
          wide,
          as.data.frame(c(list(Method = m),
                          setNames(as.list(rep(NA_real_, length(datasets))),
                                   datasets)),
                        stringsAsFactors = FALSE)
        )
      }
    }
    wide <- wide[match(METHOD_ORDER, as.character(wide$Method)), , drop = FALSE]
    wide$Method <- as.character(wide$Method)
    
    score_mat <- as.matrix(wide[, datasets, drop = FALSE])
    rownames(score_mat) <- as.character(wide$Method)
    
    rank_mat <- apply(score_mat, 2, function(x) {
      if (sum(!is.na(x)) < 2) return(rep(NA_real_, length(x)))
      if (higher_better) rank(-x, ties.method = "average", na.last = "keep")
      else               rank( x, ties.method = "average", na.last = "keep")
    })
    rownames(rank_mat) <- rownames(score_mat)
    
    kw <- compute_kendall_w(rank_mat)
    mean_rank <- rowMeans(rank_mat, na.rm = TRUE)
    # Sort methods by mean rank (best first)
    ord <- order(mean_rank, na.last = TRUE)
    list(
      clf = clf, mech = mech, rate = rate, metric = metric,
      datasets = datasets,
      rank_mat = rank_mat[ord, , drop = FALSE],
      score_mat = score_mat[ord, , drop = FALSE],
      mean_rank = mean_rank[ord],
      kw = kw,
      higher_better = higher_better
    )
  }
  
  method_tex_label <- function(m) {
    lab <- unname(METHOD_DISPLAY[as.character(m)])
    if (is.na(lab) || !nzchar(lab)) lab <- as.character(m)
    # LaTeX-safe; bold Stacking / MIE-SE for visibility in rank tables
    lab_tex <- gsub("_", "\\_", lab, fixed = TRUE)
    lab_tex <- gsub("%", "\\%", lab_tex, fixed = TRUE)
    if (as.character(m) %in% c("ImpStack_MICE", "MICE_SE"))
      return(paste0("\\textbf{", lab_tex, "}"))
    lab_tex
  }
  
  metric_tex <- function(metric) {
    switch(metric,
           Cindex   = "Harrell C-index",
           IBS      = "IBS",
           CalSlope = "calibration slope",
           metric)
  }
  
  # One LaTeX rank table: methods ordered by mean rank (#), Kendall W + p footer
  rank_block_to_latex <- function(blk) {
    if (is.null(blk)) return("")
    ds <- blk$datasets
    rm <- blk$rank_mat
    mr <- blk$mean_rank
    kw <- blk$kw
    methods <- rownames(rm)
    n_ds <- length(ds)
    
    ds_hdr <- paste(sprintf("\\textbf{\\rotatebox{45}{%s}}", ds), collapse = " & ")
    note_dir <- if (blk$higher_better) "higher score = better" else "lower score = better"
    
    lines <- c(
      "\\begin{table}[!ht]",
      "  \\centering",
      sprintf(paste0(
        "  \\caption{Ranks within \\texttt{%s} (\\emph{%s}, %s, %.0f\\%%;",
        " %s). Methods ordered by mean rank (\\#; best first).",
        " Cell rank~1 = best on that dataset; \\textbf{Mean} = average rank",
        " (lower better). Footer: Kendall's $W$ (datasets as judges) with",
        " $\\chi^2$ and $p$. $^{***}p<0.001$, $^{**}p<0.01$, $^{*}p<0.05$,",
        " ns = not significant.}"),
        blk$clf, metric_tex(blk$metric), blk$mech, blk$rate * 100, note_dir),
      sprintf("  \\label{tab:rank_%s_%s_%s_%02d}",
              tolower(blk$clf), tolower(blk$metric),
              tolower(blk$mech), as.integer(blk$rate * 100)),
      "  \\resizebox{\\linewidth}{!}{%",
      sprintf("  \\begin{tabular}{@{}cl%s@{}}",
              paste(rep("c", n_ds + 1L), collapse = "")),
      "  \\toprule",
      paste0("  \\textbf{\\#} & \\textbf{Method} & ", ds_hdr,
             " & \\textbf{Mean} \\\\"),
      "  \\midrule"
    )
    
    best_per_col <- apply(rm, 2, function(x) suppressWarnings(min(x, na.rm = TRUE)))
    best_mean <- suppressWarnings(min(mr, na.rm = TRUE))
    
    for (i in seq_along(methods)) {
      cells <- vapply(seq_along(ds), function(j) {
        v <- rm[i, j]
        if (is.na(v) || !is.finite(v)) return("--")
        s <- sprintf("%.1f", v)
        if (is.finite(best_per_col[j]) && abs(v - best_per_col[j]) < 1e-8)
          paste0("\\textbf{", s, "}") else s
      }, character(1))
      mean_s <- if (is.finite(mr[i])) {
        s <- sprintf("%.2f", mr[i])
        if (is.finite(best_mean) && abs(mr[i] - best_mean) < 1e-8)
          paste0("\\textbf{", s, "}") else s
      } else "--"
      lines <- c(lines,
                 paste0("  ", i, " & ", method_tex_label(methods[i]), " & ",
                        paste(cells, collapse = " & "), " & ", mean_s, " \\\\"))
    }
    
    w_str <- if (!is.na(kw$W)) sprintf("%.4f", kw$W) else "--"
    chi_str <- if (!is.na(kw$chi2)) sprintf("%.3f", kw$chi2) else "--"
    p_tex <- if (is.na(kw$p)) {
      "$p$ = --"
    } else if (kw$p < 0.001) {
      "$p < 0.001$"
    } else {
      sprintf("$p = %.3f$", kw$p)
    }
    df_str <- if (!is.na(kw$df)) as.character(as.integer(kw$df)) else "?"
    # # | Method | datasets... | Mean  →  footer spans datasets + Mean
    w_line <- sprintf(
      paste0("  & Kendall's $W$ & \\multicolumn{%d}{c}{",
             "$W = %s$,\\ $\\chi^2_{%s} = %s$,\\ %s\\ %s} \\\\"),
      n_ds + 1L, w_str, df_str, chi_str, p_tex, kw$signif)
    
    c(lines, "  \\midrule", w_line,
      "  \\bottomrule", "  \\end{tabular}%", "  }", "\\end{table}", "")
  }
  
  # Grand-mean wide LaTeX table (methods × rates) for one clf × mech × metric
  grand_to_latex <- function(clf, mech, metric) {
    higher_better <- identical(metric, "Cindex")
    cal_mode <- identical(metric, "CalSlope")
    sub <- grand %>%
      filter(Classifier == clf, MissMech == mech) %>%
      transmute(Method = as.character(Method), MissRate, val = .data[[metric]])
    if (nrow(sub) == 0) return("")
    
    wide <- tidyr::pivot_wider(sub, names_from = MissRate, values_from = val)
    wide <- wide[match(intersect(METHOD_ORDER, wide$Method), wide$Method), , drop = FALSE]
    rate_names <- setdiff(names(wide), "Method")
    rate_nums  <- as.numeric(rate_names)
    ord <- order(rate_nums)
    rate_names <- rate_names[ord]
    rate_nums  <- rate_nums[ord]
    if (length(rate_names) == 0) return("")
    
    hdr <- paste(sprintf("\\textbf{%.0f\\%%}", rate_nums * 100), collapse = " & ")
    mat <- as.matrix(wide[, rate_names, drop = FALSE])
    best <- apply(mat, 2, function(x) {
      if (cal_mode) {
        # closest to 1
        i <- which.min(abs(x - 1)); if (!length(i)) return(NA_real_)
        return(x[i])
      }
      if (higher_better) suppressWarnings(max(x, na.rm = TRUE))
      else               suppressWarnings(min(x, na.rm = TRUE))
    })
    
    lines <- c(
      "\\begin{table}[!ht]",
      "  \\centering",
      sprintf(paste0(
        "  \\caption{Grand-mean %s for \\texttt{%s} under %s",
        " (mean over datasets; 95\\%% CI across datasets in CSV).",
        " Best in each column in \\textbf{bold}%s.}"),
        metric_tex(metric), clf, mech,
        if (cal_mode) "; best = closest to 1" else ""),
      sprintf("  \\label{tab:grand_%s_%s_%s}",
              tolower(clf), tolower(metric), tolower(mech)),
      "  \\resizebox{\\linewidth}{!}{%",
      sprintf("  \\begin{tabular}{@{}l%s@{}}",
              paste(rep("c", length(rate_names)), collapse = "")),
      "  \\toprule",
      paste0("  \\textbf{Method} & ", hdr, " \\\\"),
      "  \\midrule"
    )
    for (i in seq_len(nrow(wide))) {
      cells <- vapply(seq_along(rate_names), function(j) {
        v <- mat[i, j]
        if (is.na(v) || !is.finite(v)) return("--")
        s <- sprintf("%.4f", v)
        if (is.finite(best[j]) && abs(v - best[j]) < 1e-10)
          paste0("\\textbf{", s, "}") else s
      }, character(1))
      lines <- c(lines,
                 paste0("  ", method_tex_label(wide$Method[i]),
                        " & ", paste(cells, collapse = " & "), " \\\\"))
    }
    c(lines, "  \\bottomrule", "  \\end{tabular}%", "  }", "\\end{table}", "")
  }
  
  METRICS_ALL <- c("Cindex", "IBS", "CalSlope")
  kendall_rows <- list()
  rank_tex_chunks <- list()
  
  cat("\n>>> Building Kendall rank tables + LaTeX...\n")
  mechs_present <- unique(as.character(results$MissMech))
  rates_by_mech <- lapply(mechs_present, function(m) {
    sort(unique(results$MissRate[as.character(results$MissMech) == m]))
  })
  names(rates_by_mech) <- mechs_present
  
  for (clf in SURV_NAMES) {
    for (mech in mechs_present) {
      for (metric in METRICS_ALL) {
        tex_body <- character(0)
        rates_use <- rates_by_mech[[mech]]
        for (rate in rates_use) {
          blk <- make_rank_block(clf, mech, rate, metric)
          if (is.null(blk)) next
          kw <- blk$kw
          mice_rank <- if ("ImpStack_MICE" %in% names(blk$mean_rank))
            unname(blk$mean_rank["ImpStack_MICE"]) else NA_real_
          kendall_rows[[length(kendall_rows) + 1L]] <- data.frame(
            Classifier = clf,
            MissMech   = mech,
            MissRate   = rate,
            Metric     = metric,
            Kendall_W  = kw$W,
            Chi2       = kw$chi2,
            df         = kw$df,
            p_value    = kw$p,
            Signif     = kw$signif,
            n_methods  = kw$n_obj,
            n_datasets = kw$m,
            ImpStack_MICE_mean_rank = mice_rank,
            stringsAsFactors = FALSE
          )
          tex_body <- c(tex_body, rank_block_to_latex(blk))
          cat(sprintf("  rank: %s | %s | %s | rate=%.3f  W=%.3f %s\n",
                      clf, metric, mech, rate,
                      ifelse(is.na(kw$W), NA, kw$W), kw$signif))
        }
        # Ranks first (primary); grand means appended (secondary / descriptive)
        tex_body <- c(tex_body, grand_to_latex(clf, mech, metric))
        if (length(tex_body) > 0) {
          fpath <- file.path(
            LATEX_DIR,
            sprintf("tables_%s_%s_%s.tex", clf, metric, mech))
          writeLines(c(latex_preamble(9), tex_body, "\\end{document}"), fpath)
        }
      }
    }
  }
  
  kendall_df <- dplyr::bind_rows(kendall_rows)
  write_csv(kendall_df, file.path(OUTPUT_DIR, "kendall_W_summary.csv"))
  write_csv(kendall_df, file.path(LATEX_DIR, "kendall_W_summary.csv"))
  
  # Explicit mean-rank CSV (includes MIE-SE / MICE_SE for every metric)
  rank_csv_rows <- list()
  for (clf in SURV_NAMES) {
    for (mech in mechs_present) {
      for (metric in METRICS_ALL) {
        for (rate in rates_by_mech[[mech]]) {
          blk <- make_rank_block(clf, mech, rate, metric)
          if (is.null(blk)) next
          rank_csv_rows[[length(rank_csv_rows) + 1L]] <- data.frame(
            Classifier = clf,
            MissMech = mech,
            MissRate = rate,
            Metric = metric,
            Method = names(blk$mean_rank),
            Display = unname(METHOD_DISPLAY[names(blk$mean_rank)]),
            MeanRank = as.numeric(blk$mean_rank),
            Kendall_W = blk$kw$W,
            stringsAsFactors = FALSE
          )
        }
      }
    }
  }
  rank_csv <- dplyr::bind_rows(rank_csv_rows)
  write_csv(rank_csv, file.path(OUTPUT_DIR, "rank_mean_all_methods.csv"))
  write_csv(rank_csv, file.path(LATEX_DIR, "rank_mean_all_methods.csv"))
  cat("Saved rank CSV (includes MICE_SE/MIE-SE): rank_mean_all_methods.csv\n")
  
  # Summary LaTeX: Kendall W across rates (ImpStack_MICE focus) for C-index
  kendall_summary_latex <- function() {
    sub <- kendall_df %>% filter(Metric == "Cindex")
    if (nrow(sub) == 0) return("")
    lines <- c(
      "\\begin{table}[!ht]",
      "  \\centering",
      "  \\caption{Kendall's coefficient of concordance $W$ for within-classifier",
      "    method rankings (Harrell C-index). Judges = datasets; objects = methods.",
      "    Also shown: mean rank of ImpStack\\_MICE (1 = best).",
      "    $^{***}p<0.001$, $^{**}p<0.01$, $^{*}p<0.05$.}",
      "  \\label{tab:kendall_W_cindex}",
      "  \\resizebox{\\linewidth}{!}{%",
      "  \\begin{tabular}{@{}llccccc@{}}",
      "  \\toprule",
      "  \\textbf{Classifier} & \\textbf{Mech} & \\textbf{10\\%} & \\textbf{20\\%} &",
      "  \\textbf{30\\%} & \\textbf{40\\%} & \\textbf{50\\%} \\\\",
      "  \\midrule"
    )
    for (clf in SURV_NAMES) {
      for (mech in MISS_MECHS) {
        cells <- sapply(MISS_RATES, function(r) {
          row <- sub %>% filter(Classifier == clf, MissMech == mech,
                                abs(MissRate - r) < 1e-6)
          if (nrow(row) == 0 || is.na(row$Kendall_W[1])) return("--")
          sprintf("%.3f%s (%.2f)",
                  row$Kendall_W[1], row$Signif[1],
                  row$ImpStack_MICE_mean_rank[1])
        })
        lines <- c(lines,
                   sprintf("  %s & %s & %s \\\\",
                           if (mech == MISS_MECHS[1]) clf else "",
                           mech,
                           paste(cells, collapse = " & ")))
      }
      if (!(clf == SURV_NAMES[length(SURV_NAMES)] &&
            mech == MISS_MECHS[length(MISS_MECHS)]))
        lines <- c(lines, "  \\midrule")
    }
    c(lines,
      "  \\bottomrule",
      "  \\multicolumn{7}{@{}p{0.95\\linewidth}@{}}{\\footnotesize",
      "    Cell format: $W$ (significance) (ImpStack\\_MICE mean rank).",
      "    Rank 1 = best.} \\\\",
      "  \\end{tabular}%", "  }", "\\end{table}", "")
  }
  
  # Master LaTeX: MAIN = ranks + deltas; SUPP = grand means (descriptive)
  delta_to_latex <- function(clf, mech, metric, baseline = REF_BASELINE) {
    if (!exists("delta_summary") || !is.data.frame(delta_summary)) return("")
    sub <- delta_summary %>%
      filter(Classifier == clf, MissMech == mech, Metric == metric,
             as.character(Method) != baseline)
    if (nrow(sub) == 0) return("")
    wide <- tidyr::pivot_wider(
      sub %>% transmute(Method = as.character(Method), MissRate,
                        val = mean_improvement),
      names_from = MissRate, values_from = val
    )
    wide <- wide[match(intersect(METHOD_ORDER, wide$Method), wide$Method), , drop = FALSE]
    rate_names <- setdiff(names(wide), "Method")
    rate_nums <- as.numeric(rate_names)
    ord <- order(rate_nums)
    rate_names <- rate_names[ord]
    rate_nums <- rate_nums[ord]
    if (length(rate_names) == 0) return("")
    hdr <- paste(sprintf("\\textbf{%.0f\\%%}", rate_nums * 100), collapse = " & ")
    mat <- as.matrix(wide[, rate_names, drop = FALSE])
    best <- apply(mat, 2, function(x) suppressWarnings(max(x, na.rm = TRUE)))
    lines <- c(
      "\\begin{table}[!ht]",
      "  \\centering",
      sprintf(paste0(
        "  \\caption{PRIMARY: mean paired improvement in %s vs %s for \\texttt{%s}",
        " under %s (positive = better than baseline; average of within-dataset deltas).",
        " Best column in \\textbf{bold}.}"),
        metric_tex(metric), baseline, clf, mech),
      sprintf("  \\label{tab:delta_%s_%s_%s}",
              tolower(clf), tolower(metric), tolower(mech)),
      "  \\resizebox{\\linewidth}{!}{%",
      sprintf("  \\begin{tabular}{@{}l%s@{}}",
              paste(rep("c", length(rate_names)), collapse = "")),
      "  \\toprule",
      paste0("  \\textbf{Method} & ", hdr, " \\\\"),
      "  \\midrule"
    )
    for (i in seq_len(nrow(wide))) {
      cells <- vapply(seq_along(rate_names), function(j) {
        v <- mat[i, j]
        if (is.na(v) || !is.finite(v)) return("--")
        s <- sprintf("%+.4f", v)
        if (is.finite(best[j]) && abs(v - best[j]) < 1e-10)
          paste0("\\textbf{", s, "}") else s
      }, character(1))
      lines <- c(lines,
                 paste0("  ", method_tex_label(wide$Method[i]),
                        " & ", paste(cells, collapse = " & "), " \\\\"))
    }
    c(lines, "  \\bottomrule", "  \\end{tabular}%", "  }", "\\end{table}", "")
  }
  
  main_lines <- c(
    latex_preamble(10),
    "\\section*{PRIMARY evaluation design}",
    paste0("Ranks and paired deltas vs \\texttt{", REF_BASELINE, "} are primary."),
    "Grand-mean absolute scores are descriptive only (see supplement).",
    "",
    "\\section*{Kendall's $W$ summary (Harrell C-index)}",
    kendall_summary_latex(),
    ""
  )
  for (clf in SURV_NAMES) {
    main_lines <- c(main_lines, sprintf("\\subsection*{%s}", clf))
    for (mech in MISS_MECHS) {
      main_lines <- c(main_lines, sprintf("\\subsubsection*{%s}", mech))
      main_lines <- c(main_lines, delta_to_latex(clf, mech, "Cindex"))
      for (rate in MISS_RATES) {
        blk <- make_rank_block(clf, mech, rate, "Cindex")
        main_lines <- c(main_lines, rank_block_to_latex(blk))
      }
    }
  }
  # IBS / CalSlope rank blocks (main metrics) — compact: one rate grid via existing files
  main_lines <- c(main_lines,
                  "\\section*{IBS and calibration: see per-metric TeX files}",
                  "\\texttt{tables\\_<Classifier\\>\\_IBS\\_<Mech\\>.tex},",
                  "\\texttt{tables\\_<Classifier\\>\\_CalSlope\\_<Mech\\>.tex}.",
                  "\\end{document}")
  writeLines(main_lines, file.path(LATEX_DIR, "main_manuscript_tables.tex"))
  
  supp_lines <- c(
    latex_preamble(10),
    "\\section*{Supplementary: grand-mean absolute scores}",
    "Descriptive only across heterogeneous datasets; not the primary estimand.",
    "Discrimination uses Harrell C-index only (Uno C not computed).",
    ""
  )
  for (clf in SURV_NAMES) {
    for (mech in MISS_MECHS) {
      for (metric in PRIMARY_METRICS) {
        supp_lines <- c(supp_lines, grand_to_latex(clf, mech, metric))
      }
    }
  }
  supp_lines <- c(supp_lines, "\\end{document}")
  writeLines(supp_lines, file.path(LATEX_DIR, "supplement_tables.tex"))
  
  # Backward-compatible master = main + pointer to supplement
  master_lines <- c(
    latex_preamble(10),
    "\\section*{PRIMARY summaries}",
    "See also \\texttt{main\\_manuscript\\_tables.tex}.",
    kendall_summary_latex(),
    "\\section*{SUPPLEMENT}",
    "See \\texttt{supplement\\_tables.tex} for grand means.",
    "\\end{document}"
  )
  writeLines(master_lines, file.path(LATEX_DIR, "all_results_tables.tex"))
  writeLines(c(latex_preamble(10), kendall_summary_latex(), "\\end{document}"),
             file.path(LATEX_DIR, "kendall_W_summary.tex"))
  
  cat(sprintf("Saved Kendall W CSV: %s\n",
              file.path(OUTPUT_DIR, "kendall_W_summary.csv")))
  cat(sprintf("Saved LaTeX folder:  %s\n", LATEX_DIR))
  cat(sprintf("  - all_results_tables.tex (master)\n"))
  cat(sprintf("  - kendall_W_summary.tex\n"))
  cat(sprintf("  - tables_<Classifier>_<Metric>_<Mech>.tex\n"))
  
}, error = function(e) {
  message(">>> LaTeX/Kendall section error (figures still saved if earlier): ",
          conditionMessage(e))
})

# Kendall heatmap (needs kendall_df from section 11b)
if (exists("kendall_df") && is.data.frame(kendall_df) && nrow(kendall_df) > 0 &&
    exists("save_plot_safe") && exists("theme_paper")) {
  tryCatch({
    kw_plot <- kendall_df %>%
      filter(Metric == "Cindex") %>%
      mutate(
        Rate_lab = factor(paste0(as.integer(MissRate * 100), "%"),
                          levels = paste0(as.integer(MISS_RATES * 100), "%")),
        W_lab = sprintf("%.2f%s", Kendall_W, Signif)
      )
    p_kw <- ggplot(kw_plot, aes(x = Rate_lab, y = Classifier, fill = Kendall_W)) +
      geom_tile(colour = "white", linewidth = 0.5) +
      geom_text(aes(label = W_lab), size = 3) +
      facet_wrap(~ MissMech) +
      scale_fill_gradientn(
        colours = c("#F7FCF5", "#74C476", "#238B45", "#00441B"),
        name = "Kendall W", limits = c(0, 1)
      ) +
      labs(title = "2D scan - Kendall's W (C-index rank concordance)",
           subtitle = "Within each classifier: agreement of method rankings across datasets",
           caption = "Objects = methods; judges = datasets",
           x = "Missingness rate", y = NULL) +
      theme_paper(11) +
      theme(panel.grid = element_blank(), legend.position = "right")
    save_plot_safe(p_kw, "07_heatmap_KendallW_2Dscan", width = 10, height = 5, dpi = 220)
  }, error = function(e) message(">>> Kendall figure skip: ", conditionMessage(e)))
}

cat("\nSaved to", OUTPUT_DIR, ":\n")
cat("  manuscript_eval/   <--- primary deltas, win rates\n")
cat("  delta_vs_baseline_summary.csv\n")
cat("  winrate_best_by_dataset.csv\n")
cat("  computation_structure.txt / .rds\n")
cat("  computation_cost_report.txt\n")
cat("  timing_per_split.csv\n")
cat("  timing_by_dataset.csv\n")
cat("  timing_summary.csv\n")
cat("  timing/   <--- copies of structure + timing\n")
cat("  impstack_raw_results.csv\n")
cat("  fallback_summary_by_method.csv / fallback_counts_by_type.csv\n")
cat("  impstack_meta_coefs_by_split.csv\n")
cat("  impstack_meta_coefs_mean.csv\n")
cat("  impstack_summary_by_dataset.csv\n")
cat("  impstack_grand_means.csv   <--- SECONDARY / supplement\n")
cat("  impstack_best_by_metric.csv\n")
cat("  impstack_best_retained_wide.csv\n")
cat("  impstack_best_by_metric_dataset.csv\n")
cat("  kendall_W_summary.csv\n")
cat("  rank_mean_all_methods.csv\n")
cat("  latex/main_manuscript_tables.tex\n")
cat("  latex/supplement_tables.tex\n")
cat("  figures/   <--- look here for PNG/PDF\n")
write_reproducibility_end(results)
cat("\nReproducibility bundle:", REPRO_DIR, "\n")
cat("Done.\n")
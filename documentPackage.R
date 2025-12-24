{
  rm(list = ls())
  options(error = NULL)

  # Package configuration (change these for different packages)
  package_name <- "strategize"
  base_path <- sprintf("~/Documents/%s-software", package_name)

  setwd(base_path)
  package_path <- file.path(base_path, package_name)

  # Get version number from DESCRIPTION
  versionNumber <- read.dcf(file.path(package_path, "DESCRIPTION"), fields = "Version")[1, 1]
  cat(sprintf("Documenting %s version %s\n", package_name, versionNumber))

  # Generate datalist for package data (if any)
  try(tools::add_datalist(package_path, force = TRUE, small.size = 1L), silent = TRUE)

  # Document package (generates .Rd files and NAMESPACE)
  devtools::document(package_path)

  # Remove old PDF manual
  pdf_file <- sprintf("./%s.pdf", package_name)
  try(file.remove(pdf_file), silent = TRUE)

  # Create new PDF manual
  system(sprintf("R CMD Rd2pdf %s", shQuote(package_path)))

  # Build tarball
  system(paste(
    shQuote(file.path(R.home("bin"), "R")),
    "CMD build --resave-data",
    shQuote(package_path)
  ))

  # Check package with CRAN standards (includes running tests)
  tarball <- sprintf("%s_%s.tar.gz", package_name, versionNumber)
  system(paste(
    shQuote(file.path(R.home("bin"), "R")),
    "CMD check --as-cran",
    shQuote(tarball)
  ))

  # Install from local source
  install.packages(package_path, repos = NULL, type = "source", force = FALSE)

  # Run tests interactively (optional, for development)
  # Tests are also run during R CMD check above
  cat("\n--- Running testthat tests ---\n")
  if (requireNamespace("testthat", quietly = TRUE)) {
    testthat::test_dir(file.path(package_path, "tests", "testthat"),
                       reporter = testthat::ProgressReporter)
  } else {
    cat("testthat not installed, skipping interactive test run\n")
  }
}

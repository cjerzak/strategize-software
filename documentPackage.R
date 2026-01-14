{
  rm(list = ls())
  options(error = NULL)
  # install.packages( "~/Documents/strategize-software/strategize",repos = NULL, type = "source",force = F);
  # strategize::build_backend()

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

  # Build vignettes
  devtools::build_vignettes(package_path)

  # Document package (generates .Rd files and NAMESPACE)
  devtools::document(package_path)

  # Remove old PDF manual
  pdf_file <- sprintf("./%s.pdf", package_name)
  try(file.remove(pdf_file), silent = TRUE)

  # Create new PDF manual
  system(sprintf("R CMD Rd2pdf %s", shQuote(package_path)))

  # Run tests (stop on failure)
  test_results <- devtools::test(package_path)
  if (any(as.data.frame(test_results)$failed > 0)) {
    stop("Tests failed! Stopping build process.")
  }
  cat("\n\U2713 All tests passed!\n\n")

  # Show object sizes in environment (for debugging memory usage)
  log(sort(sapply(ls(), function(l_) { object.size(eval(parse(text = l_))) })))

  # Check package to ensure it meets CRAN standards
  devtools::check(package_path)

  # Build tarball
  system(paste(
    shQuote(file.path(R.home("bin"), "R")),
    "CMD build --resave-data",
    shQuote(package_path)
  ))

  # Check as CRAN
  tarball <- sprintf("%s_%s.tar.gz", package_name, versionNumber)
  system(paste(
    shQuote(file.path(R.home("bin"), "R")),
    "CMD check --as-cran",
    shQuote(tarball)
  ))

  # Manual commands for reference:
  # R CMD build --resave-data ~/Documents/strategize-software/strategize
  # R CMD check --as-cran ~/Documents/strategize-software/strategize_0.0.1.tar.gz

  # Install from local source
  install.packages(package_path, repos = NULL, type = "source", force = FALSE)
  # install.packages( "~/Documents/strategize-software/strategize",repos = NULL, type = "source",force = F);
  # strategize::build_backend()
  # devtools::install_github("cjerzak/strategize-software/strategize", dependencies = TRUE) # install from github
}

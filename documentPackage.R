setwd("~/Documents/strategize-software")

package_path <- "~/Documents/strategize-software/strategize"

devtools::document(package_path)
try(file.remove(sprintf("./strategize.pdf")),T)
system(paste(shQuote(file.path(R.home("bin"), "R")),
             "CMD", "Rd2pdf", shQuote(package_path)))

#install.packages(package_path)

#install.packages( "~/Documents/optimalcausalities-software/optimalcausalities",repos = NULL, type = "source")

#install.packages("~/Library/gurobi911/mac64/R/gurobi_9.1-1_R_4.0.2.tgz",repos=NULL)

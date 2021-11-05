.phony = clean_all clean_stan clean_results clean_pdf clean_data

QUOTE_LINES = sed "s/^/'/;s/$$/'/"  # pipe this to make sure filenames are quoted

CMDSTAN_LOGS = $(shell find results/samples -type f -name "*.txt" | $(QUOTE_LINES))
STAN_OBJECT_CODE = \
  $(shell find src/stan -type f \( -not -name "*.stan" -not -name "*.md" \) \
  | $(QUOTE_LINES))
SAMPLES = $(shell find results/samples -name "*.csv" | $(QUOTE_LINES))
FAKE_DATA = $(shell find data/fake -type f -name "*.csv" | $(QUOTE_LINES))
PREPARED_DATA = $(shell find data/prepared -name "*.csv" | $(QUOTE_LINES))
INFDS = $(shell find results/infd -type f -not -name "*.md" | $(QUOTE_LINES))
LOOS = $(shell find results/loo -type f -not -name "*.md" | $(QUOTE_LINES))
JSONS = $(shell find results/input_data_json -type f -not -name "*.md" | $(QUOTE_LINES))
# Report
LATEX_FILE = report/final_report.tex
BIBLIOGRAPHY = report/bibliography.bib
PDF_FILE = report/final_report.pdf

$(PDF_FILE): $(LATEX_FILE) $(BIBLIOGRAPHY)
# The use-make option will probably help here later to enforce the dependencies properly with figures etc.
	latexmk -cd -pdf $<

clean_all: clean_stan clean_results clean_pdf clean_data

clean_data:
	$(RM) $(FAKE_DATA) $(PREPARED_DATA)

clean_stan:
	$(RM) $(CMDSTAN_LOGS) $(STAN_OBJECT_CODE)

clean_results:
	$(RM) $(SAMPLES) $(INFDS) $(LOOS) $(PLOTS) $(JSONS) $(CMDSTAN_LOGS)

clean_pdf:
	latexmk -cd -C $(latexmk -cd -C $(LATEX_FILE))

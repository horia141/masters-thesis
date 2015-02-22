all: Thesis.pdf Presentation.pdf WSOM_Paper.pdf WSOM_Presentation.pdf

Thesis.pdf: Thesis.tex Bibliography.bib UPBThesis.cls
	pdflatex Thesis.tex
	bibtex Thesis.aux
	pdflatex Thesis.tex
	pdflatex Thesis.tex

Presentation.pdf: Presentation.tex
	pdflatex Presentation.tex
	pdflatex Presentation.tex

WSOM_Paper.pdf: WSOM_Paper.tex
	pdflatex WSOM_Paper.tex

WSOM_Presentation.pdf: WSOM_Presentation.tex
	pdflatex WSOM_Presentation.tex

clean:
	rm -f *.aux
	rm -f *.bbl
	rm -f *.blg
	rm -f *.log
	rm -f *.lof
	rm -f *.lot 
	rm -f *.out 
	rm -f *.toc

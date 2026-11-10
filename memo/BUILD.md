# Build the decision memo

Open `Decision_Memo.docx` to edit the memo. The released PDF was typeset from the saved Word file, including the tables, embedded figure, and page break. Word applications may wrap individual lines differently.

~~~sh
python -m pip install python-docx reportlab
python memo/build_memo.py
~~~

To typeset an edited Word file without rebuilding the editable copy, run from the repository root:

~~~sh
python -c "from memo.build_memo import pdf_from_saved_word; pdf_from_saved_word()"
~~~


### Topic Model for NLP

***

Each script is intended to be executed in this sequence to complete the entire process :
* 001_collect_texts_from_guttenberg.ipynb
	- surveys the home page of guttenberg.org for a list of candidate book titles
	- saves the target texts to local directory ./corpus/
* 002_remove_gutenberg_licence.ipynb
	- reads each of the texts from ./corpus/ and strips guttenberg license info 
	- saves the cleaned version of texts to ./corpus_no_license/
* 003_topic_modeling_gutenberg_gutenberg.ipynb
	- reads in each text from ./coprpus_no_licence/*.txt
	- executes cleaning / normalization / topic modeling
	- eventually generates a word cloud image, which is written to directory ./plots/
	- __Note:__ normalization may require a few hours execution time for ~100+ texts corpus

***

The environment and package versions :

Linux-4.15.0-39-generic-x86_64-with-Ubuntu-16.04-xenial
('Python', '2.7.12 (default, Nov 12 2018, 14:36:49) \n[GCC 5.4.0 20160609]')
('pandas version:', u'0.23.0')
('OS', 'posix')
('Numpy', '1.15.4')
('Beautiful Soup', '4.6.3')
('Regex', '2.2.1')
('scikit-learn version', '0.19.0')
('matplotlib', '2.0.2')
('mglearn version:', '0.1.7')

***

# Simple online vector quantization methods for a data stream using remove-birth updating

These programs are my proposed simple online vector quantization methods OKRB, SOMRB, and NGRB. These online methods can efficiently quantize data and quickly adapt to data changes through remove-birth (RB) updating. In RB updating, a unit with a low win probability is removed and a new unit is born around a unit with a high win probability.  This process efficiently improves the adaptation of reference vectors to changes in data.

- OKRB (okrb.py): This is based on online k-means. This method is useful for quantizing data.
- NGRB (ngrb.py): This is based on neural gas. This method is useful for quantizing data and extracting a network from data.
- SOMRB (somrb.py):  This is based on Kohonen's SOM. This method is useful for projecting data into 2-dimensional space.

For a demonstration of quantization using these methods, see scatter.ipynb.

## References

- Kazuhisa Fujita (2023) An efficient and straightforward online quantization method for a data stream through remove-birth updating.

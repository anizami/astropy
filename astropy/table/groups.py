import numpy as np
import warnings

from itertools import izip, count

__all__ = ['TableGroups', 'ColumnGroups']


def table_group_by(table, keys):
    """
    Get groups for ``table`` on specified ``keys``.

    Parameters
    ----------
    table : `Table`
        Table to group
    keys : str, list of str, `Table`, or Numpy array
        Grouping key specifier

    Returns
    -------
    grouped_table : Table object with groups attr set accordingly
    """
    from .table import Table

    # Pre-convert string to tuple of strings, or Table to the underlying structured array
    if isinstance(keys, basestring):
        keys = (keys,)

    if isinstance(keys, (list, tuple)):
        for name in keys:
            if name not in table.colnames:
                raise ValueError('Table does not have key column {0!r}'.format(name))
            if table.masked and np.any(table[name].mask):
                raise ValueError('Missing values in key column {0!r} are not allowed'.format(name))

        keys = tuple(keys)
        table_keys = table[keys]
        grouped_by_table_cols = True  # Grouping keys are columns from the table being grouped

    elif isinstance(keys, (np.ndarray, Table)):
        table_keys = keys
        if len(table_keys) != len(table):
            raise ValueError('Input keys array length {0} does not match table length {1}'
                             .format(len(table_keys), len(table)))
        grouped_by_table_cols = False  # Grouping key(s) are external

    else:
        raise TypeError('Keys input must be string, list, tuple or numpy array, but got {0}'
                        .format(type(keys)))

    idx_sort = table_keys.argsort()
    table_keys = table_keys[idx_sort]

    # Get all keys
    diffs = np.concatenate(([True], table_keys[1:] != table_keys[:-1], [True]))
    indices = np.flatnonzero(diffs)

    # Make a new table and set the _groups to the appropriate TableGroups object.
    # Take the subset of the original keys at the indices values (group boundaries).
    out = table.__class__(table[idx_sort])
    out_keys = table_keys[indices[:-1]]
    if isinstance(out_keys, Table):
        out_keys.meta['grouped_by_table_cols'] = grouped_by_table_cols
    out._groups = TableGroups(out, indices=indices, keys=out_keys)

    return out


def column_group_by(column, keys):
    """
    Get groups for ``column`` on specified ``keys``

    Parameters
    ----------
    column : Column object
        Column to group
    keys : Table or Numpy array of same length as col
        Grouping key specifier

    Returns
    -------
    grouped_column : Column object with groups attr set accordingly
    """
    from .table import Table

    if isinstance(keys, Table):
        keys = keys._data

    if not isinstance(keys, np.ndarray):
        raise TypeError('Keys input must be numpy array, but got {0}'
                        .format(type(keys)))

    if len(keys) != len(column):
        raise ValueError('Input keys array length {0} does not match column length {1}'
                         .format(len(keys), len(column)))

    idx_sort = keys.argsort()
    keys = keys[idx_sort]

    # Get all keys
    diffs = np.concatenate(([True], keys[1:] != keys[:-1], [True]))
    indices = np.flatnonzero(diffs)

    # Make a new column and set the _groups to the appropriate ColumnGroups object.
    # Take the subset of the original keys at the indices values (group boundaries).
    out = column.__class__(column[idx_sort])
    out._groups = ColumnGroups(out, indices=indices, keys=keys[indices[:-1]])

    return out


class BaseGroups(object):
    """
    A class to represent groups within a table of heterogeneous data.

      - ``keys``: key values corresponding to each group
      - ``indices``: index values in parent table or column corresponding to group boundaries
      - ``aggregate()``: method to create new table by aggregating within groups
    """
    @property
    def parent(self):
        return self.parent_column if isinstance(self, ColumnGroups) else self.parent_table

    def __iter__(self):
        self._iter_index = 0
        return self

    def next(self):
        ii = self._iter_index
        if ii < len(self.indices) - 1:
            i0, i1 = self.indices[ii], self.indices[ii + 1]
            self._iter_index += 1
            return self.parent[i0:i1]
        else:
            raise StopIteration

    def __getitem__(self, item):
        parent = self.parent

        if isinstance(item, int):
            i0, i1 = self.indices[item], self.indices[item + 1]
            out = parent[i0:i1]
            out.groups._keys = parent.groups.keys[item]
        else:
            indices0, indices1 = self.indices[:-1], self.indices[1:]
            try:
                i0s, i1s = indices0[item], indices1[item]
            except:
                raise TypeError('Index item for groups attribute must be a slice, '
                                'numpy mask or int array')
            mask = np.zeros(len(parent), dtype=np.bool)
            # Is there a way to vectorize this in numpy?
            for i0, i1 in izip(i0s, i1s):
                mask[i0:i1] = True
            out = parent[mask]
            out.groups._keys = parent.groups.keys[item]
            out.groups._indices = np.concatenate([[0], np.cumsum(i1s - i0s)])

        return out

    def __repr__(self):
        return '<{0} indices={1}>'.format(self.__class__.__name__, self.indices)

    def __len__(self):
        return len(self.indices) - 1


class ColumnGroups(BaseGroups):
    def __init__(self, parent_column, indices=None, keys=None):
        self.parent_column = parent_column  # parent Column
        self.parent_table = parent_column.parent_table
        self._indices = indices
        self._keys = keys

    @property
    def indices(self):
        # If the parent column is in a table then use group indices from table
        if self.parent_table:
            return self.parent_table.groups.indices
        else:
            if self._indices is None:
                return np.array([0, len(self.parent_column)])
            else:
                return self._indices

    @property
    def keys(self):
        # If the parent column is in a table then use group indices from table
        if self.parent_table:
            return self.parent_table.groups.keys
        else:
            return self._keys

    def aggregate(self, func):
        i0s, i1s = self.indices[:-1], self.indices[1:]
        par_col = self.parent_column
        try:
            vals = np.array([func(par_col[i0: i1]) for i0, i1 in izip(i0s, i1s)])
        except Exception:
            raise TypeError("Cannot aggregate column '{0}'"
                            .format(par_col.name))

        out = par_col.__class__(data=vals, name=par_col.name, description=par_col.description,
                                unit=par_col.unit, format=par_col.format, meta=par_col.meta)
        return out


class TableGroups(BaseGroups):
    def __init__(self, parent_table, indices=None, keys=None):
        self.parent_table = parent_table  # parent Table
        self._indices = indices
        self._keys = keys

    @property
    def indices(self):
        if self._indices is None:
            return np.array([0, len(self.parent_table)])
        else:
            return self._indices

    def aggregate(self, func):
        """
        Aggregate each group in the Table into a single row by applying the reduction
        function ``func`` to group values in each column.

        Parameters
        ----------
        func : function
            Function that reduces an array of values to a single value

        Returns
        -------
        out : Table
            New table with the aggregated rows.
        """
        i0s, i1s = self.indices[:-1], self.indices[1:]
        out_cols = []
        parent_table = self.parent_table

        # If the table was grouped by key columns *in* the table then treat those columns
        # differently in aggregation.  In this case keys will be a Table with
        # keys.meta['grouped_by_table_cols'] == True.  Keys might not be a Table so we
        # need to handle this.
        grouped_by_table_cols = getattr(self.keys, 'meta', {}).get('grouped_by_table_cols', False)

        for col in parent_table.columns.values():
            # For key columns just pick off first in each group since they are identical
            if col.name in (self.keys.colnames if grouped_by_table_cols else ()):
                # Should just be new_col = col.take(i0s), but there is a bug in
                # MaskedColumn finalize:
                # >>> c = MaskedColumn(data=[1,2], name='a', description='a')
                # >>> print c[1:2].description
                # None
                new_col = col.__class__(data=col.take(i0s), name=col.name,
                                        description=col.description,
                                        unit=col.unit, format=col.format, meta=col.meta)
            else:
                try:
                    new_col = col.groups.aggregate(func)
                except TypeError as err:
                    warnings.warn(str(err))
                    continue

            out_cols.append(new_col)

        return parent_table.__class__(out_cols, meta=parent_table.meta)

    @property
    def keys(self):
        return self._keys

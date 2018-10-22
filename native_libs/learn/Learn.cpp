#include "Learn.h"
#include <cmath>
#include <arrow/array.h>
#include <Core/ArrowUtilities.h>
#include "SKLearn.h"
#include <variant.h>
#include <LifetimeManager.h>
#include <Analysis.h>
#include "Core/Error.h"
#include "Python/PythonInterpreter.h"

COMPILATION_UNIT_USING_NUMPY;

namespace
{

struct NPArrayBuilder
{
    size_t rows, cols;
    std::vector<double> data;

    NPArrayBuilder(size_t _rows, size_t _cols)
        : rows(_rows), cols(_cols), data(rows * cols)
    {}
    void setAt(int64_t row, int64_t col, double d)
    {
        data[row*cols+col] = d;
    }
    void setAt(int64_t row, int64_t col, int64_t i)
    {
        //if (row==0) std::cout << (double) i << std::endl;
        data[row*cols+col] = (double) i;
    }
    void setAt(int64_t row, int64_t col, std::string_view)
    {
        throw std::runtime_error("Cannot use strings with numpy array.");
    }
    void setAt(int64_t row, int64_t col, Timestamp)
    {
        throw std::runtime_error("Cannot use timestamps with numpy array.");
    }
    void setNaAt(int64_t row, int64_t col)
    {
        data[row*cols+col] = nan(" ");
    }
    pybind11::array getNPMatrix()
    {
        return pybind11::array_t<double>({rows, cols}, data.data());
    }
    pybind11::array_t<double> getNPArr()
    {
        return pybind11::array_t<double>(rows * cols, data.data());
    }
    template<typename Iterable>
    void addColumn(int columnIndex, const Iterable &iterable)
    {
        // TODO? perhaps can relax rules and fill with nulls / ignore additional values
        if(iterable.length() != rows)
            THROW("failed to add column with index {} to numpy matrix: it has {} rows while expected {}", columnIndex, iterable.length(), rows);

        int64_t rowIndex = 0;
        iterateOverGeneric(iterable,
            [&](auto&& elem)
            {
                setAt(rowIndex, columnIndex, elem);
                rowIndex++;
            },
            [&]()
            {
                setNaAt(rowIndex, columnIndex);
                rowIndex++;
            });
    }
};

auto fromC(PyObject *obj)
{
    return pybind11::reinterpret_borrow<pybind11::object>(obj);
}

auto passToC(pybind11::object obj)
{
    return obj.release().ptr();
}

} // anonymous namespace

pybind11::array tableToNpMatrix(const arrow::Table& table)
{
    NPArrayBuilder builder( table.num_rows(), table.num_columns() );
    auto cols = getColumns(table);
    int colIndex = 0;
    for (auto& col : cols)
    {
        builder.addColumn(colIndex++, *col);
    }
    auto res = builder.getNPMatrix();
    //PyObject_Print(res, stdout, 0);
    return res;
}

pybind11::array_t<double> columnToNpArr(const arrow::Column &col)
{
    NPArrayBuilder builder( col.length(), 1 );
    builder.addColumn(0, col);
    auto res = builder.getNPArr();
    //PyObject_Print(res, stdout, 0);
    return res;
}

std::shared_ptr<arrow::Column> npArrayToColumn(pybind11::array_t<double> arr, std::string name)
{
    // TODO: check that this is 1-D array
    // TODO: check that array contains doubles
    auto size = arr.shape(0);
    auto data = (const double *) arr.data();

    // TODO: check that array is contiguous
    arrow::DoubleBuilder builder;
    for(size_t i = 0; i < size; ++i)
    {
        auto value = data[i];
        if(std::isnan(value))
            builder.AppendNull();
        else
            builder.Append(value);
    }
    return toColumn(finish(builder), name);
}

void sklearn::fit(pybind11::object model, const arrow::Table &xs, const arrow::Column &y)
{
    auto xsO = tableToNpMatrix(xs);
    auto yO = columnToNpArr(y);
    sklearn::fit(model, xsO, yO);
}

double sklearn::score(pybind11::object model, const arrow::Table &xs, const arrow::Column &y)
{
    auto xsO = tableToNpMatrix(xs);
    auto yO = columnToNpArr(y);
    return sklearn::score(model, xsO, yO);
}

std::shared_ptr<arrow::Column> sklearn::predict(pybind11::object model, const arrow::Table &xs)
{
    auto xsO = tableToNpMatrix(xs);
    auto yO = sklearn::predict(model, xsO);
    auto predictedColumn = npArrayToColumn(yO, "Predictions");
    return predictedColumn;
}

std::shared_ptr<arrow::Table> sklearn::confusionMatrix(const arrow::Column &ytrue, const arrow::Column &ypred)
{
    auto c1 = columnToNpArr(ytrue);
    auto c2 = columnToNpArr(ypred);
    auto t1 = sklearn::confusion_matrix(c1, c2);
    THROW("not implemented");
}

extern "C"
{

EXPORT void toNpArr(const arrow::Table* tb, const char **outError) noexcept
{
    return TRANSLATE_EXCEPTION(outError)
    {
        tableToNpMatrix(*tb);
    };
}

EXPORT void freePyObj(PyObject* o, const char **outError) noexcept
{
    return TRANSLATE_EXCEPTION(outError)
    {
        Py_XDECREF(o);
    };
}

EXPORT PyObject* newLogisticRegression(double C, const char **outError) noexcept
{
    return TRANSLATE_EXCEPTION(outError)
    {
        return passToC(sklearn::newLogisticRegression(C));
    };
}

EXPORT PyObject* newLinearRegression(const char **outError) noexcept
{
    return TRANSLATE_EXCEPTION(outError)
    {
        return passToC(sklearn::newLinearRegression());
    };
}

EXPORT void fit(PyObject* model, const arrow::Table *xs, const arrow::Column *y, const char **outError) noexcept
{
    return TRANSLATE_EXCEPTION(outError)
    {
        return sklearn::fit(fromC(model), *xs, *y);
    };
}

EXPORT double score(PyObject* model, const arrow::Table* xs, const arrow::Column* y, const char **outError) noexcept
{
    return TRANSLATE_EXCEPTION(outError)
    {
        return sklearn::score(fromC(model), *xs, *y);
    };
}

EXPORT arrow::Column* predict(PyObject* model, const arrow::Table* xs, const char **outError) noexcept
{
    return TRANSLATE_EXCEPTION(outError)
    {
        auto ret = sklearn::predict(fromC(model), *xs);
        return LifetimeManager::instance().addOwnership(ret);
    };
}

EXPORT arrow::Table* confusionMatrix(const arrow::Column* ytrue, const arrow::Column* ypred, const char **outError) noexcept
{
    return TRANSLATE_EXCEPTION(outError)
    {
        auto ret = sklearn::confusionMatrix(*ytrue, *ypred);
        return LifetimeManager::instance().addOwnership(ret);
    };
}

EXPORT arrow::Table* oneHotEncode(const arrow::Column* col, const char **outError) noexcept
{
    return TRANSLATE_EXCEPTION(outError)
    {
        std::unordered_map<std::string_view, int> valIndexes;
        int lastIndex = 0;
        iterateOver<arrow::Type::STRING>(*col,
            [&](auto &&elem)
            {
                if (valIndexes.find(elem)==valIndexes.end()) 
                    valIndexes[elem] = lastIndex++;
            },
            []() { });
        std::vector<arrow::DoubleBuilder> builders(valIndexes.size());
        iterateOver<arrow::Type::STRING>(*col,
            [&](auto &&elem)
            {
                int ind = valIndexes[elem];
                for (int i = 0; i<builders.size(); i++)
                {
                    builders[i].Append(i==ind ? 1 : 0);
                }
            },
            [&]()
            {
                for (int i = 0; i<builders.size(); i++)
                {
                    builders[i].Append(0);
                }
            });
        std::vector<PossiblyChunkedArray> arrs;
        for (auto& bldr : builders)
        {
            arrs.push_back(finish(bldr));
        }
        std::vector<std::string> names(valIndexes.size());
        for (auto& item : valIndexes)
        {
            names[item.second] = col->name()+": "+std::string(item.first);
        }
        auto t = tableFromArrays(arrs, names);
        return LifetimeManager::instance().addOwnership(std::move(t));
    };
}

} // extern "C"

sklearn::interpreter& sklearn::interpreter::get()
{
    static sklearn::interpreter ctx;
    return ctx;
}

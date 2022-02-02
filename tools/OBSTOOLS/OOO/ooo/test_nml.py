#!/usr/bin/env python2.7
import unittest
import nml

def version(vn):
    # Converts module version string to int tuple as PEP-440
    return tuple(map(int, vn.split(".")))

HEADER = """
!! =====================================
!! An example FORTRAN 90 namelist
!! =====================================
&namex
"""
NOHEADER = "&namex\n"
LOGIC = " ln_test = .TRUE. ! Comment\n"
NUMERIC = " nn_off_idx = 1 2 3\n"
NOSPACE = " nn_off_idx=1 2 3\n"
NEGATIVE = " nn_off_idx = -1 -2 -3\n"
LIST = " off_files = 'a.nc' 'b.nc' 'c.nc' ! Comment\n"
STRING = ' contact = "example@nowhere.com" ! Comment\n'
COMMENT = ' ! x = y '
FOOTER = "/\n"
NOFOOTER = "/"

class TestNamelist(unittest.TestCase):
    def setUp(self):
        self.logic_text = "\n".join([HEADER, LOGIC, FOOTER])
        self.numeric_text = "\n".join([HEADER, NUMERIC, FOOTER])
        self.nospace_text = "\n".join([HEADER, NOSPACE, FOOTER])
        self.neg_numeric_text = "\n".join([HEADER, NEGATIVE, FOOTER])
        self.list_text = "\n".join([HEADER, LIST, FOOTER])
        self.mix_text = "\n".join([HEADER, LIST, NUMERIC, FOOTER])
        self.string_text = "\n".join([HEADER, STRING, FOOTER])
        self.comment_text = "\n".join([HEADER, COMMENT, FOOTER])
        self.no_trail_text = "\n".join([HEADER, NUMERIC, LIST, NOFOOTER])

    def test_should_handle_footer(self):
        data = nml.variables(self.no_trail_text)
        truth = {"nn_off_idx": "1 2 3",
                 "off_files": "'a.nc' 'b.nc' 'c.nc'"}
        self.assertDictEqual(data, truth)

    def test_mix_data(self):
        data = nml.variables(self.mix_text)
        truth = {"nn_off_idx": "1 2 3",
                 "off_files": "'a.nc' 'b.nc' 'c.nc'"}
        self.assertDictEqual(data, truth)

    def test_comment_data(self):
        data = nml.variables(self.comment_text)
        truth = {}
        self.assertDictEqual(data, truth)

    def test_logical_data(self):
        data = nml.variables(self.logic_text)
        truth = {"ln_test": ".TRUE."}
        self.assertDictEqual(data, truth)

    def test_numeric_data(self):
        data = nml.variables(self.numeric_text)
        truth = {"nn_off_idx": "1 2 3"}
        self.assertDictEqual(data, truth)

    def test_nospace_data(self):
        data = nml.variables(self.nospace_text)
        truth = {"nn_off_idx": "1 2 3"}
        self.assertDictEqual(data, truth)

    def test_negative_numeric_data(self):
        data = nml.variables(self.neg_numeric_text)
        truth = {"nn_off_idx": "-1 -2 -3"}
        self.assertDictEqual(data, truth)

    def test_string_data(self):
        data = nml.variables(self.string_text)
        truth = {"contact": '"example@nowhere.com"'}
        self.assertDictEqual(data, truth)

    def test_list_data(self):
        data = nml.variables(self.list_text)
        truth = {"off_files": "'a.nc' 'b.nc' 'c.nc'"}
        self.assertDictEqual(data, truth)

    def test_replace_variable_comment(self):
        FIXTURE = " x = 'y' ! comment \n"
        RESULT = " x = 'z' ! comment \n"
        text = nml.replace(FIXTURE, {"x": "z"})
        self.assertEqual(text, RESULT)

    def test_replace_variable_no_comment(self):
        FIXTURE = " x = 'y' \n"
        RESULT = " x = 'z' \n"
        text = nml.replace(FIXTURE, {"x": "z"})
        self.assertEqual(text, RESULT)

    def test_replace_variable_no_space(self):
        FIXTURE = " x='y' \n"
        RESULT = " x='z' \n"
        text = nml.replace(FIXTURE, {"x": "z"})
        self.assertEqual(text, RESULT)

    def test_replace_variable_no_space_comment(self):
        FIXTURE = " x='y' ! comment \n"
        RESULT = " x='z' ! comment \n"
        text = nml.replace(FIXTURE, {"x": "z"})
        self.assertEqual(text, RESULT)

    @unittest.skipIf(version(nml.__version__) < (1, 0),
                     "Not implemented in this version")
    def test_multiline_variable_replace(self):
        # As a result of a code review
        FIXTURE = " x = 1, \n     2"
        RESULT =  " x = 3, \n     4"
        text = nml.replace(FIXTURE, {"x": [3,4]})
        self.assertEqual(text, RESULT)

class TestCombinedNamelists(unittest.TestCase):
    def setUp(self):
        self.minimal = """&namone
/"""
        self.blanks = """
&namone
/
&namtwo
/
"""
        self.contents = """
&namone
Content 1
/
&namtwo
Content 2
/
"""
        self.contents_slash = """
&namone
Content 1 ! Y/N
/
&namtwo
Content 2
/
"""
    def test_should_select_name_from_minimal_namelist(self):
        data = nml.namelists(self.minimal)
        result = {"namone": "&namone\n/"}
        self.assertDictEqual(data, result)

    def test_should_select_names(self):
        data = nml.namelists(self.blanks)
        result = {"namone": "&namone\n/",
                  "namtwo": "&namtwo\n/"}
        self.assertDictEqual(data, result)

    def test_should_select_contents(self):
        data = nml.namelists(self.contents)
        result = {"namone": "&namone\nContent 1\n/",
                  "namtwo": "&namtwo\nContent 2\n/"}
        self.assertDictEqual(data, result)

    def test_should_select_contents_with_forward_slash(self):
        data = nml.namelists(self.contents_slash)
        result = {"namone": "&namone\nContent 1 ! Y/N\n/",
                  "namtwo": "&namtwo\nContent 2\n/"}
        self.assertDictEqual(data, result)


class TestToString(unittest.TestCase):
    def setUp(self):
        self.char_list = ["a.nc", "b.nc", "c.nc"]
        self.num_list = [1, 3, 7]
        self.char = "foo@bar.com"
        self.num = 10
        self.mixed_list = ["foo.nc", -1, True]
        self.mixed_str_list = map(nml.quote, ["foo.nc", "-1", ".TRUE."])

    def test_should_format_mixed_list(self):
        data = nml.tostring(self.mixed_list)
        result = "'foo.nc', -1, .TRUE."
        self.assertEqual(data, result)

    def test_should_format_numeric_list(self):
        data = nml.tostring(self.num_list)
        result = "1 3 7"
        self.assertEqual(data, result)

    def test_should_format_character_list(self):
        data = nml.tostring(self.char_list)
        result = "'a.nc', 'b.nc', 'c.nc'"
        self.assertEqual(data, result)

    def test_should_format_strings(self):
        data = nml.tostring(self.char)
        result = "'%s'" % (self.char,)
        self.assertEqual(data, result)

    def test_should_format_numbers(self):
        data = nml.tostring(self.num)
        result = str(self.num)
        self.assertEqual(data, result)

    def test_should_not_format_numeric_string(self):
        input = "3.14159"
        self.assertEqual(nml.tostring(input), input)

    def test_should_format_logicals(self):
        data = nml.tostring(True)
        result = ".TRUE."
        self.assertEqual(data.upper(), result)

    def test_should_not_format_string_of_list_data(self):
        for input in ["1 2 3", "1, 2, 3", ".TRUE. .FALSE."]:
            case = nml.tostring(input)
            self.assertEqual(case, input)

    def test_should_treat_mixed_numeric_character_data_as_character(self):
        case = nml.tostring(self.mixed_str_list)
        result = "'foo.nc', '-1', '.TRUE.'"
        self.assertEqual(case, result)

class TestUpdateNamelist(unittest.TestCase):
    def setUp(self):
        self.empty = """
&namone
/
"""
        self.single = """
&namone
   x = 'y'
/
"""
        self.single_update = """
&namone
   x = 'z'
/
"""

    def test_should_append_new_variable_to_namelist(self):
        trial = nml.update("namone", self.empty, {"x": "'y'"})
        self.assertEqual(trial, self.single)

    def test_should_update_existing_variables(self):
        trial = nml.update("namone", self.single, {"x": "'z'"})
        self.assertEqual(trial, self.single_update)

if __name__ == '__main__':
    unittest.main()


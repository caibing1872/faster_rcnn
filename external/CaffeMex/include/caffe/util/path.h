#ifndef _PATH_H_
#define _PATH_H_
#include <stdlib.h>
#include <string>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

using std::string;


static void _split_whole_name(const char *whole_name, char *fname, char *ext)
{
	char *p_ext=rindex(const_cast<char*>(whole_name), '.');
	if (NULL != p_ext)
	{
		strcpy(ext, p_ext);
		snprintf(fname, p_ext - whole_name + 1, "%s", whole_name);
	}
	else
	{
		ext[0] = '\0';
		strcpy(fname, whole_name);
	}
}

void _splitpath(const char *path, char *drive, char *dir, char *fname, char *ext)
{
	char *p_whole_name;

	drive[0] = '\0';
	if (NULL == path)
	{
		dir[0] = '\0';
		fname[0] = '\0';
		ext[0] = '\0';
		return;
	}

	if ('/' == path[strlen(path)])
	{
		strcpy(dir, path);
		fname[0] = '\0';
		ext[0] = '\0';
		return;
	}

	p_whole_name = rindex(const_cast<char*>(path), '/');
	if (NULL != p_whole_name)
	{
		p_whole_name++;
		_split_whole_name(p_whole_name, fname, ext);

		snprintf(dir, p_whole_name - path, "%s", path);
	}
	else
	{
		_split_whole_name(path, fname, ext);
		dir[0] = '\0';
	}
}



class CPath
{

public:
	static string GetFileName(string strPath)
	{
		char pDrive[PATH_MAX], pDir[PATH_MAX], pFileName[PATH_MAX], pExt[PATH_MAX];
		char pOutput[PATH_MAX];
		_splitpath(strPath.c_str(), pDrive, pDir, pFileName, pExt);
		sprintf(pOutput, "%s%s", pFileName, pExt);
		return string(pOutput);
	}


	static string GetFileNameWithoutExtension(string strPath)
	{
		char pDrive[PATH_MAX], pDir[PATH_MAX], pFileName[PATH_MAX], pExt[PATH_MAX];
		_splitpath(strPath.c_str(), pDrive, pDir, pFileName, pExt);
		return string(pFileName);
	}


	static string GetDirectoryName(string strPath)
	{
		char pDrive[PATH_MAX], pDir[PATH_MAX], pFileName[PATH_MAX], pExt[PATH_MAX];
		char pOutput[PATH_MAX];
		_splitpath(strPath.c_str(), pDrive, pDir, pFileName, pExt);
		sprintf(pOutput, "%s%s", pDrive, pDir);
		return string(pOutput);
	}


	static string GetExtension(string strPath)
	{
		char pDrive[PATH_MAX], pDir[PATH_MAX], pFileName[PATH_MAX], pExt[PATH_MAX];
		_splitpath(strPath.c_str(), pDrive, pDir, pFileName, pExt);
		return string(pExt);
	}
};

#endif

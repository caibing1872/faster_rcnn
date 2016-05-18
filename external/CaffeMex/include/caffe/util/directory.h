#include <unistd.h>
#include <vector>
#include <string>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>

using std::vector;
using std::string;

class CDirectory
{
public:
    static vector<string> GetFiles(const char *strFolder, const char *strFilter, bool bAllDirectories)
    {
        vector<string> vec = GetFilesInOneFolder(strFolder, strFilter);
        if (bAllDirectories)
        {
            vector<string> vecSubFolders = GetDirectories(strFolder, "*", true);
            for (size_t i = 0; i < vecSubFolders.size(); i++)
            {
                vector<string> vecFiles = GetFilesInOneFolder(vecSubFolders[i].c_str(), strFilter);
                for (size_t j = 0; j < vecFiles.size(); j++)
                    vec.push_back(vecFiles[j]);
            }
        }
        return vec;
    }

    static vector<string> GetDirectories(const char *strFolder, const char *strFilter, bool bAllDirectories)
    {
        vector<string> vec = GetDirectoryInOnFolder(strFolder, strFilter);
        if (vec.size() == 0)
            return vec;
        if (bAllDirectories)
        {
            vector<string> vecSubFolder;
            for (size_t i = 0; i < vec.size(); i++)
            {
                vector<string> vecSub = GetDirectories(vec[i].c_str(), strFilter, bAllDirectories);
                for (size_t j = 0; j < vecSub.size(); j++)
                {
                    vecSubFolder.push_back(vecSub[j]);
                }
            }
            for (size_t i = 0; i < vecSubFolder.size(); i++)
                vec.push_back(vecSubFolder[i]);
        }
        return vec;
    }

    static string GetCurrentDirectory()
    {
        char strPath[PATH_MAX];
        getcwd(strPath, PATH_MAX);
        return string(strPath);
    }


    static bool Exist(const char *strPath)
    {
        return (access(strPath, 0) == 0);
    }


    static bool CreateDirectory(const char *strPath)
    {
        if (Exist(strPath))
            return false;
        char strFolder[PATH_MAX] = {0};
        size_t len = strlen(strPath);
        for (size_t i = 0; i <= len; i++)
        {
            if (strPath[i] == '\\' || strPath[i] == '/' || strPath[i] == '\0')
            {
                if (!Exist(strFolder))
                {
                    if(mkdir(strFolder, S_IRWXU) == 0)
                        return false;
                }
            }
            strFolder[i] = strPath[i];
        }
        return true;
    }


private:
    static vector<string> GetFilesInOneFolder(const char *strFolder, const char *strFilter)
    {
        vector<string> vec;
        char strFile[PATH_MAX] = {'\0'};
        sprintf(strFile, "%s\\%s", strFolder, strFilter);

        DIR *d;
        struct dirent *file;

        if(!(d = opendir(strFolder)))
        {
            return vec;
        }
        while((file = readdir(d)) != NULL)
        {
            if(strncmp(file->d_name, ".", 1) == 0)
                continue;
            char strName[PATH_MAX];
            sprintf(strName, "%s\\%s", strFolder, file->d_name);
            vec.push_back(strName);
        }
        closedir(d);
        return vec;
    }

    static vector<string> GetDirectoryInOnFolder(const char *strFolder, const char *strFilter)
    {
        vector<string> vec;
        char strFile[PATH_MAX] = {'\0'};
        sprintf(strFile, "%s\\%s", strFolder, strFilter);

        DIR *d;
        struct dirent *file;
        struct stat sb;

        if(!(d = opendir(strFolder)))
        {
            return vec;
        }
        while((file = readdir(d)) != NULL)
        {
            if(strncmp(file->d_name, ".", 1) == 0)
                continue;

            if(stat(file->d_name, &sb) >= 0 && S_ISDIR(sb.st_mode) )
            {
                char strName[PATH_MAX];
                sprintf(strName, "%s\\%s", strFolder, file->d_name);
                vec.push_back(strName);
            }

        }
        closedir(d);
        return vec;
    }
};



